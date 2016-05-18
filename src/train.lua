-- Track accuracy
opt.lastAcc = opt.lastAcc or 0
opt.bestAcc = opt.bestAcc or 0
-- We save snapshots of the best model only when evaluating on the full validation set
trackBest = (opt.validIters * opt.validBatch == ref.valid.nsamples)

-- The dimensions of 'final predictions' are defined by the opt.task file
-- This allows some flexibility for post-processing of the network output
preds = torch.Tensor(ref.valid.nsamples, unpack(predDim))

-- We also save the raw output of the network (in this case heatmaps)
if type(outputDim[1]) == "table" then predHMs = torch.Tensor(ref.valid.nsamples, unpack(outputDim[#outputDim]))
else predHMs = torch.Tensor(ref.valid.nsamples, unpack(outputDim)) end

-- Model parameters
param, gradparam = model:getParameters()

-- Main processing step
function step(tag)
    local avgLoss, avgAcc = 0.0, 0.0
    local output, err, idx
    local r = ref[tag]
    local function evalFn(x) return criterion.output, gradparam end

    if tag == 'train' then
        print("==> Starting epoch: " .. epoch .. "/" .. (opt.nEpochs + opt.epochNumber - 1))
        model:training()
        set = 'train'
        isTesting = false -- Global flag
    else
        if tag == 'predict' then print("==> Generating predictions...") end
        model:evaluate()
        set = 'valid'
        isTesting = true
    end

    for i,sample in loader[set]:run() do

        xlua.progress(i, r.iters)
        local input, label = unpack(sample)

        if opt.GPU ~= -1 then
            -- Convert to CUDA
            input = applyFn(function (x) return x:cuda() end, input)
            label = applyFn(function (x) return x:cuda() end, label)
        end

        -- Do a forward pass and calculate loss
        local output = model:forward(input)
        local err = criterion:forward(output, label)

        -- Training: Do backpropagation and optimization
        if tag == 'train' then
            model:zeroGradParameters()
            model:backward(input, criterion:backward(output, label))
            optfn(evalFn, param, optimState)

        -- Validation: Get flipped output
        else
            output = applyFn(function (x) return x:clone() end, output)
            local flip_ = customFlip or flip
            local shuffleLR_ = customShuffleLR or shuffleLR
            local flippedOut = model:forward(flip_(input))
            flippedOut = applyFn(function (x) return flip_(shuffleLR_(x)) end, flippedOut)
            output = applyFn(function (x,y) return x:add(y):div(2) end, output, flippedOut)

        end

        -- Synchronize with GPU
        if opt.GPU ~= -1 then cutorch.synchronize() end

        -- If we're generating predictions, save output
        if tag == 'predict' or (tag == 'valid' and trackBest) then
            if type(outputDim[1]) == "table" then
                -- If we're getting a table of heatmaps, save the last one
                predHMs:sub(i,i+r.batchsize-1):copy(output[#output])
            else
                predHMs:sub(i,i+r.batchsize-1):copy(output)
            end
            if postprocess then preds:sub(i,i+r.batchsize-1):copy(postprocess(set,i,output)) end
        end

        -- Calculate accuracy
        local acc = accuracy(output, label)
        avgLoss = avgLoss + err
        avgAcc = avgAcc + acc
    end

    avgLoss = avgLoss / r.iters
    avgAcc = avgAcc / r.iters

    local epochStep = torch.floor(ref.train.nsamples / (r.iters * r.batchsize * 2))
    if tag == 'train' and epoch % epochStep == 0 then
        if avgAcc - opt.lastAcc < opt.threshold then
            isFinished = true --Training has plateaued
        end
        opt.lastAcc = avgAcc
    end

    -- Print and log some useful performance metrics
    print(string.format("      %s : Loss: %.7f Acc: %.4f"  % {set, avgLoss, avgAcc}))
    if r.log then
        r.log:add{
            ['epoch     '] = string.format("%d" % epoch),
            ['loss      '] = string.format("%.6f" % avgLoss),
            ['acc       '] = string.format("%.4f" % avgAcc),
            ['LR        '] = string.format("%g" % optimState.learningRate)
        }
    end

    if tag == 'train' and opt.snapshot ~= 0 and epoch % opt.snapshot == 0 then
        -- Take an intermediate training snapshot
        model:clearState()
        torch.save(paths.concat(opt.save, 'model_' .. epoch .. '.t7'), model)
        torch.save(paths.concat(opt.save, 'optimState.t7'), optimState)
    elseif tag == 'valid' and trackBest and avgAcc > opt.bestAcc then
        -- A new record validation accuracy has been hit, save the model and predictions
        predFile = hdf5.open(opt.save .. '/best_preds.h5', 'w')
        predFile:write('heatmaps', predHMs)
        if postprocess then predFile:write('preds', preds) end
        predFile:close()
        model:clearState()
        torch.save(paths.concat(opt.save, 'best_model.t7'), model)
        torch.save(paths.concat(opt.save, 'optimState.t7'), optimState)
        opt.bestAcc = avgAcc
    elseif tag == 'predict' then
        -- Save final predictions
        predFile = hdf5.open(opt.save .. '/preds.h5', 'w')
        predFile:write('heatmaps', predHMs)
        if postprocess then predFile:write('preds', preds) end
        predFile:close()
    end
end

function train() step('train') end
function valid() step('valid') end
function predict() step('predict') end

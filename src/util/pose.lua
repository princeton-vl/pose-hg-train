-- Update dimension references to account for intermediate supervision
ref.predDim = {dataset.nJoints,5}
ref.outputDim = {}
criterion = nn.ParallelCriterion()
for i = 1,opt.nStack do
    ref.outputDim[i] = {dataset.nJoints, opt.outputRes, opt.outputRes}
    criterion:add(nn[opt.crit .. 'Criterion']())
end

-- Function for data augmentation, randomly samples on a normal distribution
local function rnd(x) return math.max(-2*x,math.min(2*x,torch.randn(1)[1]*x)) end

-- Code to generate training samples from raw images
function generateSample(set, idx)
    local img = dataset:loadImage(idx)
    local pts, c, s = dataset:getPartInfo(idx)
    local r = 0

    if set == 'train' then
        -- Scale and rotation augmentation
        s = s * (2 ^ rnd(opt.scale))
        r = rnd(opt.rotate)
        if torch.uniform() <= .6 then r = 0 end
    end

    local inp = crop(img, c, s, r, opt.inputRes)
    local out = torch.zeros(dataset.nJoints, opt.outputRes, opt.outputRes)
    for i = 1,dataset.nJoints do
        if pts[i][1] > 1 then -- Checks that there is a ground truth annotation
            drawGaussian(out[i], transform(torch.add(pts[i],1), c, s, r, opt.outputRes), 1)
        end
    end

    if set == 'train' then
        -- Flipping and color augmentation
        if torch.uniform() < .5 then
            inp = flip(inp)
            out = shuffleLR(flip(out))
        end
        inp[1]:mul(torch.uniform(0.6,1.4)):clamp(0,1)
        inp[2]:mul(torch.uniform(0.6,1.4)):clamp(0,1)
        inp[3]:mul(torch.uniform(0.6,1.4)):clamp(0,1)
    end

    return inp,out
end

-- Load in a mini-batch of data
function loadData(set, idxs)
    if type(idxs) == 'table' then idxs = torch.Tensor(idxs) end
    local nsamples = idxs:size(1)
    local input,label

    for i = 1,nsamples do
        local tmpInput,tmpLabel
        tmpInput,tmpLabel = generateSample(set, idxs[i])
        tmpInput = tmpInput:view(1,unpack(tmpInput:size():totable()))
        tmpLabel = tmpLabel:view(1,unpack(tmpLabel:size():totable()))
        if not input then
            input = tmpInput
            label = tmpLabel
        else
            input = input:cat(tmpInput,1)
            label = label:cat(tmpLabel,1)
        end
    end

    if opt.nStack > 1 then
        -- Set up label for intermediate supervision
        local newLabel = {}
        for i = 1,opt.nStack do newLabel[i] = label end
        return input,newLabel
    else
        return input,label
    end
end

function postprocess(set, idx, output)
    local tmpOutput
    if type(output) == 'table' then tmpOutput = output[#output]
    else tmpOutput = output end
    local p = getPreds(tmpOutput)
    local scores = torch.zeros(p:size(1),p:size(2),1)

    -- Very simple post-processing step to improve performance at tight PCK thresholds
    for i = 1,p:size(1) do
        for j = 1,p:size(2) do
            local hm = tmpOutput[i][j]
            local pX,pY = p[i][j][1], p[i][j][2]
            scores[i][j] = hm[pY][pX]
            if pX > 1 and pX < opt.outputRes and pY > 1 and pY < opt.outputRes then
               local diff = torch.Tensor({hm[pY][pX+1]-hm[pY][pX-1], hm[pY+1][pX]-hm[pY-1][pX]})
               p[i][j]:add(diff:sign():mul(.25))
            end
        end
    end
    p:add(0.5)

    -- Transform predictions back to original coordinate space
    local p_tf = torch.zeros(p:size())
    for i = 1,p:size(1) do
        _,c,s = dataset:getPartInfo(idx[i])
        p_tf[i]:copy(transformPreds(p[i], c, s, opt.outputRes))
    end
    
    return p_tf:cat(p,3):cat(scores,3)
end

function accuracy(output,label)
    if type(output) == 'table' then
        return heatmapAccuracy(output[#output],label[#output],nil,dataset.accIdxs)
    else
        return heatmapAccuracy(output,label,nil,dataset.accIdxs)
    end
end

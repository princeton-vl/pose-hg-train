-- Get prediction coordinates
predDim = {nParts,2}

criterion = nn.ParallelCriterion()
for i = 1,opt.nStack do criterion:add(nn.MSECriterion()) end

-- Code to generate training samples from raw images.
function generateSample(set, idx)
    local pts = annot[set]['part'][idx]
    local c = annot[set]['center'][idx]
    local s = annot[set]['scale'][idx]
    local img = image.load(opt.dataDir .. '/images/' .. annot[set]['images'][idx])

    -- For single-person pose estimation with a centered/scaled figure
    local inp = crop(img, c, s, 0, opt.inputRes)
    local out = torch.zeros(nParts, opt.outputRes, opt.outputRes)
    for i = 1,nParts do
        if pts[i][1] > 0 then -- Checks that there is a ground truth annotation
            drawGaussian(out[i], transform(torch.add(pts[i],1), c, s, 0, opt.outputRes), 1)
        end
    end

    return inp,out
end

function preprocess(input, label)
    if opt.nStack > 1 then
        local newLabel = {}
        for i = 1,opt.nStack do newLabel[i] = label end
        return input,newLabel
    else
        return input,label
    end
end

function postprocess(set, idx, output)
    local p,tmpOutput
    if opt.nStack > 1 then tmpOutput = output[#output]
    else tmpOutput = output end
    p = getPreds(tmpOutput)

    -- Very simple post-processing step to improve performance at tight PCK thresholds
    for i = 1,p:size(1) do
        for j = 1,p:size(2) do
            local hm = tmpOutput[i][j]
            local pX,pY = p[i][j][1], p[i][j][2]
            if pX > 1 and pX < opt.outputRes and pY > 1 and pY < opt.outputRes then
               local diff = torch.Tensor({hm[pY][pX+1]-hm[pY][pX-1], hm[pY+1][pX]-hm[pY-1][pX]})
               p[i][j]:add(diff:sign():mul(.25))
            end
        end
    end
    p:add(-0.5)
    
    return p           
end

function accuracy(output,label)
    local jntIdxs = {mpii={1,2,3,4,5,6,11,12,15,16},flic={2,3,5,6,7,8}}
    if opt.nStack > 1 then
        return heatmapAccuracy(output[#output],label[#output],nil,jntIdxs[opt.dataset])
    else
        return heatmapAccuracy(output,label,nil,jntIdxs[opt.dataset])
    end
end

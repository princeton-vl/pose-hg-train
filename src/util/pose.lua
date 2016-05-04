-- Get prediction coordinates
predDim = {nParts,2}

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
            drawGaussian(out[i], transform(torch.add(pts[i],1), c, s, 0, opt.outputRes), 2)
        end
    end

    return inp,out
end

-- function preprocess(input, label)
--     require 'image'
--     w = image.display{image=input,win=w}
--     w2 = image.display{image=label:view(label:size(1)*label:size(2),label:size(3),label:size(4)),win=w2}
--     return input, label
-- end

function postprocess(set, idx, output)
    -- Return predictions in the heatmap coordinate space
    -- The evaluation code will apply the transformation back to the original image space
    -- (Though we could also do it here)
    local preds = getPreds(output)
    return preds
end

function accuracy(output,label)
    -- Only care about accuracy across the most difficult joints
    local jntIdxs = {mpii={1,2,3,4,5,6,11,12,15,16},flic={2,3,5,6,7,8}}
    return heatmapAccuracy(output,label,nil,jntIdxs[opt.dataset])
end

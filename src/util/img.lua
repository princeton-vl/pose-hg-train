-------------------------------------------------------------------------------
-- Coordinate transformation
-------------------------------------------------------------------------------

function getTransform(center, scale, rot, res)
    local h = 200 * scale
    local t = torch.eye(3)

    -- Scaling
    t[1][1] = res / h
    t[2][2] = res / h

    -- Translation
    t[1][3] = res * (-center[1] / h + .5)
    t[2][3] = res * (-center[2] / h + .5)

    -- Rotation
    if rot ~= 0 then
        rot = -rot
        local r = torch.eye(3)
        local ang = rot * math.pi / 180
        local s = math.sin(ang)
        local c = math.cos(ang)
        r[1][1] = c
        r[1][2] = -s
        r[2][1] = s
        r[2][2] = c
        -- Need to make sure rotation is around center
        local t_ = torch.eye(3)
        t_[1][3] = -res/2
        t_[2][3] = -res/2
        local t_inv = torch.eye(3)
        t_inv[1][3] = res/2
        t_inv[2][3] = res/2
        t = t_inv * r * t_ * t
    end

    return t
end

function transform(pt, center, scale, rot, res, invert)
    local pt_ = torch.ones(3)
    pt_[1],pt_[2] = pt[1]-1,pt[2]-1

    local t = getTransform(center, scale, rot, res)
    if invert then
        t = torch.inverse(t)
    end
    local new_point = (t*pt_):sub(1,2):add(1e-4)

    return new_point:int():add(1)
end

-------------------------------------------------------------------------------
-- Cropping
-------------------------------------------------------------------------------

function crop(img, center, scale, rot, res)
    local ul = transform({1,1}, center, scale, 0, res, true)
    local br = transform({res+1,res+1}, center, scale, 0, res, true)


    local pad = math.floor(torch.norm((ul - br):float())/2 - (br[1]-ul[1])/2)
    if rot ~= 0 then
        ul = ul - pad
        br = br + pad
    end

    local newDim,newImg,ht,wd

    if img:size():size() > 2 then
        newDim = torch.IntTensor({img:size(1), br[2] - ul[2], br[1] - ul[1]})
        newImg = torch.zeros(newDim[1],newDim[2],newDim[3])
        ht = img:size(2)
        wd = img:size(3)
    else
        newDim = torch.IntTensor({br[2] - ul[2], br[1] - ul[1]})
        newImg = torch.zeros(newDim[1],newDim[2])
        ht = img:size(1)
        wd = img:size(2)
    end

    local newX = torch.Tensor({math.max(1, -ul[1] + 2), math.min(br[1], wd+1) - ul[1]})
    local newY = torch.Tensor({math.max(1, -ul[2] + 2), math.min(br[2], ht+1) - ul[2]})
    local oldX = torch.Tensor({math.max(1, ul[1]), math.min(br[1], wd+1) - 1})
    local oldY = torch.Tensor({math.max(1, ul[2]), math.min(br[2], ht+1) - 1})

    if newDim:size(1) > 2 then
        newImg:sub(1,newDim[1],newY[1],newY[2],newX[1],newX[2]):copy(img:sub(1,newDim[1],oldY[1],oldY[2],oldX[1],oldX[2]))
    else
        newImg:sub(newY[1],newY[2],newX[1],newX[2]):copy(img:sub(oldY[1],oldY[2],oldX[1],oldX[2]))
    end

    if rot ~= 0 then
        newImg = image.rotate(newImg, rot * math.pi / 180, 'bilinear')
        if newDim:size(1) > 2 then
            newImg = newImg:sub(1,newDim[1],pad,newDim[2]-pad,pad,newDim[3]-pad)
        else
            newImg = newImg:sub(pad,newDim[1]-pad,pad,newDim[2]-pad)
        end
    end

    newImg = image.scale(newImg,res,res)
    return newImg
end

function two_point_crop(img, s, pt1, pt2, pad, res)
    local center = (pt1 + pt2) / 2
    local scale = math.max(20*s,torch.norm(pt1 - pt2)) * .007
    scale = scale * pad
    local angle = math.atan2(pt2[2]-pt1[2],pt2[1]-pt1[1]) * 180 / math.pi - 90
    return crop(img, center, scale, angle, res)
end

-------------------------------------------------------------------------------
-- Non-maximum Suppression
-------------------------------------------------------------------------------

-- Set up max network for NMS
nms_window_size = 3
nms_pad = (nms_window_size - 1)/2
maxlayer = nn.Sequential()
if cudnn then
    maxlayer:add(cudnn.SpatialMaxPooling(nms_window_size, nms_window_size,1,1, nms_pad, nms_pad))
    maxlayer:cuda()
else
    maxlayer:add(nn.SpatialMaxPooling(nms_window_size, nms_window_size,1,1, nms_pad,nms_pad))
end
maxlayer:evaluate()

function local_maxes(hm, n, c, s, hm_idx)
    hm = torch.Tensor(1,16,64,64):copy(hm):float()
    if hm_idx then hm = hm:sub(1,-1,hm_idx,hm_idx) end
    local hm_dim = hm:size()
    local max_out
    -- First do nms
    if cudnn then
        local hmCuda = torch.CudaTensor(1, hm_dim[2], hm_dim[3], hm_dim[4])
        hmCuda:copy(hm)
        max_out = maxlayer:forward(hmCuda)
        cutorch.synchronize()
    else
        max_out = maxlayer:forward(hm)
    end

    local nms = torch.cmul(hm, torch.eq(hm, max_out:float()):float())[1]
    -- Loop through each heatmap retrieving top n locations, and their scores
    local pred_coords = torch.Tensor(hm_dim[2], n, 2)
    local pred_scores = torch.Tensor(hm_dim[2], n)
    for i = 1, hm_dim[2] do
        local nms_flat = nms[i]:view(nms[i]:nElement())
        local vals,idxs = torch.sort(nms_flat,1,true)
        for j = 1,n do
            local pt = {idxs[j] % 64, torch.ceil(idxs[j] / 64) }
            pred_coords[i][j] = transform(pt, c, s, 0, 64, true)
            pred_scores[i][j] = vals[j]
        end
    end
    return pred_coords, pred_scores
end

-------------------------------------------------------------------------------
-- Draw gaussian
-------------------------------------------------------------------------------

function drawGaussian(img, pt, sigma)
    -- Draw a 2D gaussian
    -- Check that any part of the gaussian is in-bounds
    local ul = {math.floor(pt[1] - 3 * sigma), math.floor(pt[2] - 3 * sigma)}
    local br = {math.floor(pt[1] + 3 * sigma), math.floor(pt[2] + 3 * sigma)}
    -- If not, return the image as is
    if (ul[1] > img:size(2) or ul[2] > img:size(1) or br[1] < 1 or br[2] < 1) then return img end
    -- Generate gaussian
    local size = 6 * sigma + 1
    local g = image.gaussian(size) -- , 1 / size, 1)
    -- Usable gaussian range
    local g_x = {math.max(1, -ul[1]), math.min(br[1], img:size(2)) - math.max(1, ul[1]) + math.max(1, -ul[1])}
    local g_y = {math.max(1, -ul[2]), math.min(br[2], img:size(1)) - math.max(1, ul[2]) + math.max(1, -ul[2])}
    -- Image range
    local img_x = {math.max(1, ul[1]), math.min(br[1], img:size(2))}
    local img_y = {math.max(1, ul[2]), math.min(br[2], img:size(1))}
    assert(g_x[1] > 0 and g_y[1] > 0)
    img:sub(img_y[1], img_y[2], img_x[1], img_x[2]):add(g:sub(g_y[1], g_y[2], g_x[1], g_x[2]))
    img[img:gt(1)] = 1
    return img
end

function drawLine(img,pt1,pt2,width,val,doGauss)
    val = val or 1
    doGauss = doGauss or 1
    pt1 = torch.Tensor(pt1)
    pt2 = torch.Tensor(pt2)
    m = torch.dist(pt1,pt2)
    dy = (pt2[2] - pt1[2])/m
    dx = (pt2[1] - pt1[1])/m
    for j = 1,width do
        start_pt1 = torch.Tensor({pt1[1] + (-width/2 + j-1)*dy,
                                  pt1[2] - (-width/2 + j-1)*dx})
        start_pt1:ceil()
        for i = 1,torch.ceil(m) do
            y_idx = torch.ceil(start_pt1[2]+dy*i)
            x_idx = torch.ceil(start_pt1[1]+dx*i)
            if doGauss == 1 and j == torch.ceil(width/2) and (i == 1 or i == torch.ceil(m)) then
                img = drawGaussian(img, {x_idx,y_idx}, width/3)
            end
            if y_idx - 1 > 0 and x_idx -1 > 0 and y_idx < img:size(1) and x_idx < img:size(2) then
                img:sub(y_idx-1,y_idx,x_idx-1,x_idx):fill(val)
            end
        end 
    end
    img[img:gt(1)] = 1
end

-------------------------------------------------------------------------------
-- Flipping functions
-------------------------------------------------------------------------------

function shuffleLR(x)
    local dim
    if x:nDimension() == 4 then
        dim = 2
    else
        assert(x:nDimension() == 3)
        dim = 1
    end

    local matchedParts
    if opt.dataset == 'mpii' then
        matchedParts = {
            {1,6},   {2,5},   {3,4},
            {11,16}, {12,15}, {13,14}
        }
    elseif opt.dataset == 'flic' then
        matched_parts = {
            {1,4}, {2,5}, {3,6}, {7,8}, {9,10}
        }
    elseif opt.dataset == 'lsp' then
        matched_parts = {
            {1,6}, {2,5}, {3,4}, {7,12}, {8,11}, {9,10}
        }
    end

    for i = 1,#matchedParts do
        local idx1, idx2 = unpack(matchedParts[i])
        local tmp = x:narrow(dim, idx1, 1):clone()
        x:narrow(dim, idx1, 1):copy(x:narrow(dim, idx2, 1))
        x:narrow(dim, idx2, 1):copy(tmp)
    end

    return x
end

function flip(x)
    require 'image'
    local y = torch.FloatTensor(x:size())
    for i = 1, x:size(1) do
        image.hflip(y[i], x[i]:float())
    end
    return y:typeAs(x)
end

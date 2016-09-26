local M = {}
Dataset = torch.class('pose.Dataset',M)

function Dataset:__init()
    self.nJoints = 16
    self.accIdxs = {1,2,3,4,5,6,11,12,15,16}
    self.flipRef = {{1,6},   {2,5},   {3,4},
                    {11,16}, {12,15}, {13,14}}
    -- Pairs of joints for drawing skeleton
    self.skeletonRef = {{1,2,1},    {2,3,1},    {3,7,1},
                        {4,5,2},    {4,7,2},    {5,6,2},
                        {7,9,0},    {9,10,0},
                        {13,9,3},   {11,12,3},  {12,13,3},
                        {14,9,4},   {14,15,4},  {15,16,4}}

    local annot = {}
    local tags = {'index','person','imgname','part','center','scale',
                  'normalize','torsoangle','visible','multi','istrain'}
    local a = hdf5.open(paths.concat(projectDir,'data/mpii/annot.h5'),'r')
    for _,tag in ipairs(tags) do annot[tag] = a:read(tag):all() end
    a:close()
    annot.index:add(1)
    annot.person:add(1)
    annot.part:add(1)

    -- Index reference
    if not opt.idxRef then
        local allIdxs = torch.range(1,annot.index:size(1))
        opt.idxRef = {}
        opt.idxRef.test = allIdxs[annot.istrain:eq(0)]
        opt.idxRef.train = allIdxs[annot.istrain:eq(1)]

        -- Set up training/validation split
        local perm = torch.randperm(opt.idxRef.train:size(1)):long()
        opt.idxRef.valid = opt.idxRef.train:index(1, perm:sub(1,opt.nValidImgs))
        opt.idxRef.train = opt.idxRef.train:index(1, perm:sub(opt.nValidImgs+1,-1))

        torch.save(opt.save .. '/options.t7', opt)
    end

    self.annot = annot
    self.nsamples = {train=opt.idxRef.train:numel(),
                     valid=opt.idxRef.valid:numel(),
                     test=opt.idxRef.test:numel()}

    -- For final predictions
    opt.testIters = self.nsamples.test
    opt.testBatch = 1
end

function Dataset:size(set)
    return self.nsamples[set]
end

function Dataset:getPath(idx)
    return paths.concat(opt.dataDir,'images',ffi.string(self.annot.imgname[idx]:char():data()))
end

function Dataset:loadImage(idx)
    return image.load(self:getPath(idx))
end

function Dataset:getPartInfo(idx)
    local pts = self.annot.part[idx]:clone()
    local c = self.annot.center[idx]:clone()
    local s = self.annot.scale[idx]
    -- Small adjustment so cropping is less likely to take feet out
    c[2] = c[2] + 15 * s
    s = s * 1.25
    return pts, c, s
end

function Dataset:normalize(idx)
    return self.annot.normalize[idx]
end

return M.Dataset


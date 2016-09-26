local M = {}
Dataset = torch.class('pose.Dataset',M)

function Dataset:__init()
    self.nJoints = 16
    self.accIdxs = {1,2,3,4,5,6,11,12,15,16}
    self.flipRef = {{1,6},   {2,5},   {3,4},
                    {11,16}, {12,15}, {13,14}}

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
end

function Dataset:size(set)
    return self.nsamples[set]
end

function Dataset:getPath(idx)
    return paths.concat(opt.dataDir,'mpii/images',ffi.string(self.annot.imgname[idx]:char():data()))
end

function Dataset:loadImage(idx)
    return image.load(self:getPath(idx))
end

function Dataset:getPartInfo(idx)
    return self.annot.part[idx], self.annot.center[idx], self.annot.scale[idx]
end

function Dataset:normalize(idx)
    return self.annot.normalize[idx]
end

return M.Dataset

    -- Pairs of joints for drawing skeleton (1=left, 2=right)
    -- self.skeletonRef = {
    --         {1,2,1},      {2,3},      {3,7},
    --         {4,5},      {4,7},      {5,6},
    --         {7,9},      {9,10},
    --         {14,9},     {11,12},    {12,13},
    --         {13,9},     {14,15},    {15,16}}

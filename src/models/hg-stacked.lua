paths.dofile('layers/Residual.lua')

local function hourglass(n, numIn, numOut, inp)
    -- Upper branch
    local up1 = Residual(numIn,256)(inp)
    local up2 = Residual(256,256)(up1)
    local up4 = Residual(256,numOut)(up2)

    -- Lower branch
    local pool = nnlib.SpatialMaxPooling(2,2,2,2)(inp)
    local low1 = Residual(numIn,256)(pool)
    local low2 = Residual(256,256)(low1)
    local low5 = Residual(256,256)(low2)
    local low6
    if n > 1 then
        low6 = hourglass(n-1,256,numOut,low5)
    else
        low6 = Residual(256,numOut)(low5)
    end
    local low7 = Residual(numOut,numOut)(low6)
    local up5 = nn.SpatialUpSamplingNearest(2)(low7)

    -- Bring two branches together
    return nn.CAddTable()({up4,up5})
end

local function lin(numIn,numOut,inp)
    -- Apply 1x1 convolution, no stride, no padding
    local l_ = nnlib.SpatialConvolution(numIn,numOut,1,1,1,1,0,0)(inp)
    return nnlib.ReLU(true)(nn.SpatialBatchNormalization(numOut)(l_))
end

function createModel()

    local inp = nn.Identity()()

    -- Initial processing of the image
    local cnv1_ = nnlib.SpatialConvolution(3,64,7,7,2,2,3,3)(inp)           -- 128
    local cnv1 = nnlib.ReLU(true)(nn.SpatialBatchNormalization(64)(cnv1_))
    local r1 = Residual(64,128)(cnv1)
    local pool = nnlib.SpatialMaxPooling(2,2,2,2)(r1)                       -- 64
    local r4 = Residual(128,128)(pool)
    local r5 = Residual(128,128)(r4)
    local r6 = Residual(128,256)(r5)

    -- First hourglass
    local hg1 = hourglass(4,256,512,r6)

    -- Linear layers to produce first set of predictions
    local l1 = lin(512,512,hg1)
    local l2 = lin(512,256,l1)

    -- First predicted heatmaps
    local out1 = nnlib.SpatialConvolution(256,outputDim[1][1],1,1,1,1,0,0)(l2)
    local out1_ = nnlib.SpatialConvolution(outputDim[1][1],256+128,1,1,1,1,0,0)(out1)

    -- Concatenate with previous linear features
    local cat1 = nn.JoinTable(2)({l2,pool})
    local cat1_ = nnlib.SpatialConvolution(256+128,256+128,1,1,1,1,0,0)(cat1)
    local int1 = nn.CAddTable()({cat1_,out1_})

    -- Second hourglass
    local hg2 = hourglass(4,256+128,512,int1)

    -- Linear layers to produce predictions again
    local l3 = lin(512,512,hg2)
    local l4 = lin(512,512,l3)

    -- Output heatmaps
    local out2 = nnlib.SpatialConvolution(512,outputDim[2][1],1,1,1,1,0,0)(l4)

    -- Final model
    local model = nn.gModule({inp}, {out1,out2})

    return model

end

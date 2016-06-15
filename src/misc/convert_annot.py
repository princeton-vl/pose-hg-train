import h5py
import numpy as np
import sys
import mpii

keys = ['index','person','imgname','center','scale','part','visible','normalize','torsoangle']
annot = {k:[] for k in keys}
dotrain = True

# Get image filenames
imgnameRef = mpii.annot['annolist'][0][0][0]['image'][:]

for idx in xrange(mpii.nimages):
    print "\r",idx,
    sys.stdout.flush()

    if mpii.istrain(idx) == dotrain:
        for person in xrange(mpii.numpeople(idx)):
            c,s = mpii.location(idx,person)
            if not c[0] == -1:
                # Adjust center/scale slightly to avoid cropping limbs
                # (in hindsight this should have been done in the Torch code...)
                c[1] += 15 * s
                s *= 1.25

                # Part annotations and visibility
                coords = np.zeros((16,2))
                vis = np.zeros(16)
                for part in xrange(16):
                   coords[part],vis[part] = mpii.partinfo(idx,person,part)

                # Add info to annotation list
                annot['index'] += [idx]
                annot['person'] += [person]
                annot['imgname'] += [str(imgnameRef[idx][0][0][0][0])]
                annot['center'] += [c]
                annot['scale'] += [s]
                annot['part'] += [coords]
                annot['visible'] += [vis]
                annot['normalize'] += [mpii.normalization(idx,person)]
                annot['torsoangle'] += [mpii.torsoangle(idx,person)]

print ""

with h5py.File('mpii-annot-train.h5','w') as f:
    f.attrs['name'] = 'mpii'
    for k in keys:
        if not k == 'imgname': annot[k] = np.array(annot[k])
        f[k] = annot[k]


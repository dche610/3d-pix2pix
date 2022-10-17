import ants


def main():
    fi = ants.image_read('preprocessed_MNI152NLin2009aSym.nii.gz')
    mi = ants.image_read('0_raw_inpaint.nii')
    mytx = ants.registration(fixed=fi, moving=mi, type_of_transform = 'SyN')
    # ants.image_write(mytx['warpedmovout'], 'r001s001_registered_inpaint.nii')

    mi2 = ants.image_read('0.nii')
    mytx2 = ants.registration(fixed=fi, moving=mi2, type_of_transform = 'SyN')
    ants.image_write(mytx2['warpedmovout'], '0_lesion.nii')

    learned_register = ants.apply_transforms(fixed=fi, moving=mi2, transformlist=mytx['fwdtransforms'])
    ants.image_write(learned_register, '0_learnt.nii')


if __name__ == '__main__':
    main()
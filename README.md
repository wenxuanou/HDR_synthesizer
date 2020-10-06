# HDR_synthesizer
For 15663 computational photography, merge several LDR images into single HDR image. Has additional program for tonemapping, color correction, and noise calibration.

## HDRjpeg.py ##
- Performs HDR merging and luminance evaluation on JPG images
- Functions:
	### downSample(img,N)
	- Downsample image
	- input: 
		• img: RGB image to be downsampled
		• N: downsample factor
	- output: downsampled image
	
	### weight(imgVal,w_type,Zmin,Zmax,t_expo=None, isReg=False,isLinear=False)
	- Function that perform pixel weighting
	- input:
		• imgVal: image pixel value, can be matrix
		• w_type: weight type: uniform/tent/gauss/photon
		• Zmin/Zmax: range of pixel value
		• t_expo: exposure time for photon weighting
		• isReg: input true if is weighting regularization term
		• isLinear: input true if is weighting in linearization
	- output:
		• img_w: weighted pixel value

	### W_uniform(imgVal,Zmin,Zmax)
	- Uniform weighting
	- input: 
		• imgVal: image pixel value, can be matrix
		• Zmin/Zmax: range of pixel value
	- output:
		• img_w: weighted pixel value

	### W_tent(imgVal,Zmin,Zmax)
	- Tent weighting 
	- input: 
		• imgVal: image pixel value, can be matrix
		• Zmin/Zmax: range of pixel value
	- output:
		• img_w: weighted pixel value

	### W_gauss(imgVal,Zmin,Zmax)
	- Gaussian weighting 
	- input: 
		• imgVal: image pixel value, can be matrix
		• Zmin/Zmax: range of pixel value
	- output:
		• img_w: weighted pixel value

	### W_photon(imgVal,t_expo,Zmin,Zmax)
	- Photon weighting
	- input: 
		• imgVal: image pixel value, can be matrix
		• t_expo: exposure time
		• Zmin/Zmax: range of pixel value
	- output:
		• img_w: weighted pixel value


	### gammaEncode(img_linear)
	- Gamma Encoding
	- input:
		• img_linear: linear image
	- output:
		• img_gamma: gamma encoded image

	### loadImg(imgNames,N)
	- Load images from system
	- input:
		• imgNames: array that stores the image file names
		• N: downsample factor
	- output:
		• img_down_flat: downsampled, flattened image stack
		• img_full_flat: full size flattened image stack
		• log_t_expo: logarithm of exposture time of each image
		• W_/ H_: size of downsampled image

	### linearization(img_down_flat,img_full_flat,log_t_expo,lamb,weight_type,W_,H_)
	- Function that performs linearization
	- input:
		• img_down_flat: downsampled image stack
		• img_full_flat: full size flattened image stack
		• log_t_expo: logarithm of exposture time of each image
		• lamb: lambda value
		• weight_type: weight type: uniform/tent/gauss/photon
		• W_/ H_: size of downsampled image
	- output:
		• img_linear: linearized image stack

	### HDR(img_flat, log_t_expo, weight_type, merge_type, W, H)
	- Function that performs HDR merging
	- input:
		• img_flat: flatten image stack
		• log_t_expo: logarithm of exposture time of each image
		• weight_type: weight type: uniform/tent/gauss/photon
		• merge_type: merge type: linear/log
		• W/ H: size of full size image
	- output:
		• img_HDR: HDR image

	### loadColorCheker()
	- load color checker for evaluation
	- input:
	- output:
		• colorchecker: matrix that stores color checker values
		• patchLoc: the neutral grey patch location

	### getImgPatchLum(img_HDR)
	- Function that selects luminance of patches in image
	- input:
		• img_HDR: the image to choose from
	- output:
		• img_patch_lum: the luminance of selected patches


## HDRraw.py ##
- Performs HDR merging and luminance evaluation on RAW (.tiff) images
- Functions:
	### downSample(img,N)
	- Downsample image
	- input: 
		• img: RGB image to be downsampled
		• N: downsample factor
	- output: downsampled image

	### weight(imgVal,w_type,Zmin,Zmax,t_expo=None,isReg=False)
	- Function that perform pixel weighting
	- input:
		• imgVal: image pixel value, can be matrix
		• w_type: weight type: uniform/tent/gauss/photon
		• Zmin/Zmax: range of pixel value
		• t_expo: exposure time for photon weighting
		• isReg: input true if is weighting regularization term
	- output:
		• img_w: weighted pixel value

	### W_uniform(imgVal,Zmin,Zmax)
	- Uniform weighting
	- input: 
		• imgVal: image pixel value, can be matrix
		• Zmin/Zmax: range of pixel value
	- output:
		• img_w: weighted pixel value

	### W_tent(imgVal,Zmin,Zmax)
	- Tent weighting 
	- input: 
		• imgVal: image pixel value, can be matrix
		• Zmin/Zmax: range of pixel value
	- output:
		• img_w: weighted pixel value

	### W_gauss(imgVal,Zmin,Zmax)
	- Gaussian weighting 
	- input: 
		• imgVal: image pixel value, can be matrix
		• Zmin/Zmax: range of pixel value
	- output:
		• img_w: weighted pixel value

	### W_photon(imgVal,t_expo,Zmin,Zmax)
	- Photon weighting
	- input: 
		• imgVal: image pixel value, can be matrix
		• t_expo: exposure time
		• Zmin/Zmax: range of pixel value
	- output:
		• img_w: weighted pixel value

	### gammaEncode(img_linear)
	- Gamma Encoding
	- input:
		• img_linear: linear image
	- output:
		• img_gamma: gamma encoded image

	### loadImg(imgNames,N)
	- Load images from system
	- input:
		• imgNames: array that stores the image file names
		• N: downsample factor
	- output:
		• img_down_flat: downsampled, flattened image stack
		• img_full_flat: full size flattened image stack
		• log_t_expo: logarithm of exposture time of each image
		• W_/ H_: size of downsampled image

	### HDR(img_flat, log_t_expo, weight_type, merge_type, W, H)
	- Function that performs HDR merging
	- input:
		• img_flat: flatten image stack
		• log_t_expo: logarithm of exposture time of each image
		• weight_type: weight type: uniform/tent/gauss/photon
		• merge_type: merge type: linear/log
		• W/ H: size of full size image
	- output:
		• img_HDR: HDR image

	### loadColorCheker()
	- load color checker for evaluation
	- input:
	- output:
		• colorchecker: matrix that stores color checker values
		• patchLoc: the neutral grey patch location

	### getImgPatchLum(img_HDR)
	- Function that selects luminance of patches in image
	- input:
		• img_HDR: the image to choose from
	- output:
		• img_patch_lum: the luminance of selected patches

## myHDR.py ##
- Perform HDR merging on my images, contains noise calibration and noise-optimal weight
- Functions:
	** Almost identical to HDRraw.py and HDRjpeg.py
	- added “optimal” weight type
	- added isNoiseCalibrate, true if want to perform noise calibration


## Tonemapping.py ##
- Performs tone mapping on HDR image
- Functions:
	### gammaEncode(img_linear)
	- Gamma Encoding
	- input:
		• img_linear: linear image
	- output:
		• img_gamma: gamma encoded image

	### tonemapping(img_HDR,tonemap_type,K,B,pixelNum,epsilon)
	- Tonemapping function
	- input:
		• img_HDR: the HDR image for tone mapping
		• tonemap_type: tone mapping type: all_channel/luminance
		• K: key value
		• B: burn value
		• pixelNum: the number of pixels in input image
		• epsilon: the small value to avoid error
	- output:
		• img_TM: the tonemapped image


## ColorCorrection.py ##
- Performs color correction and white balance on HDR image
- Functions:
	### loadColorCheker()
	- load color checker for evaluation
	- input:
	- output:
		• colorchecker: matrix that stores color checker values

	### gammaEncode(img_linear)
	- Gamma Encoding
	- input:
		• img_linear: linear image
	- output:
		• img_gamma: gamma encoded image

	### getImgPatchLum(img_HDR)
	- Function that selects 24 colorchecker patches in image
	- input:
		• img_HDR: the image to choose from
	- output:
		• img_patch: selected patches in image

	### manualBalance(img_rgb, subImg)
	- Function that performs white balance based on sub image
	- input:
		• img_rgb: image to be white balanced
		• subImg: a white reference patch in img_rgb
	- output:
		• img_rgb: balanced image










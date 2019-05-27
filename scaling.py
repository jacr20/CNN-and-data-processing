'''SCALING THE IMAGES PRE-CNN'''

from PIL import Image

FilenameWIMP = '/home/joel/Documents/XENON-ML/Figures/WIMPSIM_CNN'
FilenameER = '/home/joel/Documents/XENON-ML/Figures/ERSIM_CNN'

for i in range (0,500):
    if i<10:
        imageWIMP = Image.open(FilenameWIMP + "/000000_00000%spng.png" % i)   
        scaled_imageWIMP = imageWIMP.resize((100,50),Image.ANTIALIAS)
        scaled_imageWIMP.save("/home/joel/Documents/XENON-ML/Figures/ScaledSIM1_CNN/WIMP%s_Scaled.png" % (i),quality=95)
        imageER = Image.open(FilenameER + "/000000_00000%spng.png" % i)   
        scaled_imageER = imageER.resize((100,50),Image.ANTIALIAS)
        scaled_imageER.save("/home/joel/Documents/XENON-ML/Figures/ScaledSIM1_CNN/ER%s_Scaled.png" % (i),quality=95)
        #i += 1

    elif 10<=i<100:
        imageWIMP = Image.open(FilenameWIMP + "/000000_0000%spng.png" % i)   
        scaled_imageWIMP = imageWIMP.resize((100,50),Image.ANTIALIAS)
        scaled_imageWIMP.save("/home/joel/Documents/XENON-ML/Figures/ScaledSIM1_CNN/WIMP%s_Scaled.png" % (i),quality=95)
        imageER = Image.open(FilenameER + "/000000_0000%spng.png" % i)   
        scaled_imageER = imageER.resize((100,50),Image.ANTIALIAS)
        scaled_imageER.save("/home/joel/Documents/XENON-ML/Figures/ScaledSIM1_CNN/ER%s_Scaled.png" % (i),quality=95)
        #i += 1
    
    else:
        imageWIMP = Image.open(FilenameWIMP + "/000000_000%spng.png" % i)   
        scaled_imageWIMP = imageWIMP.resize((100,50),Image.ANTIALIAS)
        scaled_imageWIMP.save("/home/joel/Documents/XENON-ML/Figures/ScaledSIM1_CNN/WIMP%s_Scaled.png" % (i),quality=95)
        imageER = Image.open(FilenameER + "/000000_000%spng.png" % i)   
        scaled_imageER = imageER.resize((100,50),Image.ANTIALIAS)
        scaled_imageER.save("/home/joel/Documents/XENON-ML/Figures/ScaledSIM1_CNN/ER%s_Scaled.png" % (i),quality=95)
        

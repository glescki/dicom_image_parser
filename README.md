HC Dataset Parser

Simple parser of DICOM files from the dataset to jpeg images

Just run the following commando:

```
python parser.py <Path of hc_dataset>
```

The script will read each patient .dcm file and create a folder in the same
directory as of the script, the new directory will contain all slices from each
patient .dcm in order.


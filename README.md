# Dehazing Cost Volume for Deep Multi-view Stereo in Scattering Media
This is the code of the extended version of our ACCV2020 paper "Dehazing Cost Volume for Deep Multi-view Stereo in Scattering Media".
If you use this code, please cite:

```
@article{fujimura20a,
	author = {Y. Fujimura and M. Sonogashira and M. Iiyama},
	title = {Dehazing Cost Volume for Deep Multi-view Stereo in Scattering Media},
	journal = {Asian Conference on Computer Vision (ACCV)},
	year = 2020
}

@article{fujimura20b,
	author = {Y. Fujimura and M. Sonogashira and M. Iiyama},
	title = {Dehazing Cost Volume for Deep Multi-view Stereo in Scattering Media with Scattering Parameters Estimation},
	journal = {TBA},
	year = 2020
}
```

## Download trained models
Our trained models can be downloaded as follows:

```$ bash download.sh ```

## Sample code
You can get sample result by

```$ python test.py --dcv_mvsnet_checkpoint model/dehazing_mvs_net.pth --airlight_checkpoint model/airlight_estimator.pth```


## Sample result
![result](result.png)

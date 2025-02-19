# Heatmap error distance 

![heatmap example](example.png)

This is the code that was used to generate the heatmap results for [Diff-DOPE](https://arxiv.org/abs/2310.00463). If you are using BOP annotation you should be able to use the code. There is an example folder that is self contained. 

```
python heatmap.py --opencv --bop --overlay --path_json_gt example/scene_gt.json --path_json_gu example/diff_dope.json --objs_folder example/models/ --contour --spp 100
```

This code base needs a special version of NViSII, which is  downloaded and installed by the previous step, but you can always download the wheel manually [here](https://www.dropbox.com/s/m85v7ts981xs090/nvisii-1.2.dev47%2Bgf122b5b.72-cp36-cp36m-manylinux2014_x86_64.whl?dl=0).
This updated NViSII mainly add support to render background as an alpha mask when exporting png files. You are going to need it for contour.  


This should produce the image above. Use `--max_distance` to change the interval, it is expressed in cm.  

## How to cite
If you are using this code in your research, please cite this as follow, 
```
@misc{tremblay2023diffdope,
      title={Diff-DOPE: Differentiable Deep Object Pose Estimation}, 
      author={Jonathan Tremblay and Bowen Wen and Valts Blukis and Balakumar Sundaralingam and Stephen Tyree and Stan Birchfield},
      year={2023},
      eprint={2310.00463},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

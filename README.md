### PointAttN


### GraspNet
python move_npz_to_npy.py  #includes filtering the pcl
python contact_graspnet_pytorch/inference.py --np_path=npy_files/*.npy --forward_passes=5 --z_range=[0.2,1.1]​
python add_completion_fields.py ​

​
### Visualize and filter grasps ​

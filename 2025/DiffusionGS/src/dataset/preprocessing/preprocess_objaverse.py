import objaverse
import multiprocessing

uids = objaverse.load_uids()
print(f"length of uids : {len(uids)}")
print(f"type of uids : {type(uids)}")
annotations = objaverse.load_annotations(uids)
processes = multiprocessing.cpu_count()
print(f"the number of processes : {processes}")
objaverse_objects = objaverse.load_objects(uids=uids, download_processes=processes)

# objaverse
# - streaming download
# - preprocess (center and scale)
# - sample random viewpoint (N samples)
#   > View Sampler
#   > RPPC


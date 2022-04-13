# Import everything needed to edit video clips
from moviepy.editor import *

# loading video dsa gfg intro video
# getting subclip as video is large
# adding margin to the video

map_elites_mv = "logs/classic_miracle_rogue_strat_deck/to_plot/2021-07-14_21-15-31_Distributed_MAP-Elites_Classic_Miracle_Rogue_show/metrics/elites_archive/heatmap/heatmap_MAP-Elites.avi"

dsa_me_mv = "logs/classic_miracle_rogue_strat_deck/to_plot/2021-07-14_21-15-29_Surrogated_MAP-Elites_FullyConnectedNN_Classic_Miracle_Rogue_show/metrics/elites_archive/heatmap/heatmap_DSA-ME.avi"


if __name__ == '__main__':
    clip1 = VideoFileClip(dsa_me_mv)

    clip2 = VideoFileClip(map_elites_mv)

    # clips list
    clips = [clip1, clip2]

    # stacking clips
    final = clips_array([clips])

    # showing final clip
    # final.ipython_display(width=480)

    final.write_gif("archives.gif")

mkdir -p logs
python -u -m kneel.inference.app --lc_snapshot_path snapshots_release/lext-devbox_2019_07_14_16_04_41 \
       --hc_snapshot_path snapshots_release/lext-devbox_2019_07_14_19_25_40 \
       --refine True --mean_std_path snapshots_release/mean_std.npy \
      --deploy True --device cpu --port 5000 --logs logs/kneel-cpu.log

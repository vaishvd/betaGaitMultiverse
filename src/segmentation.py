def find_segment(events_df, start_marker, end_marker):

    start = events_df[events_df["value"] == start_marker]
    end = events_df[events_df["value"] == end_marker]

    if start.empty or end.empty:
        raise RuntimeError("Segment markers not found")

    start_time = float(start["onset"].values[0])
    end_time = float(end["onset"].values[0])

    return start_time, end_time


def crop_raw(raw, start_time, end_time, buffer=5):
    crop_start = max(0, start_time - buffer)
    crop_end = min(raw.times[-1], end_time + buffer)

    raw_seg = raw.copy().crop(tmin=crop_start, tmax=crop_end)

    return raw_seg, crop_start, crop_end
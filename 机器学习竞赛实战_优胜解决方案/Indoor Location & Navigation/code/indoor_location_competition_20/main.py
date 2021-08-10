import json
import os
from pathlib import Path


import numpy as np

from compute_f import split_ts_seq, compute_step_positions
from io_f import read_data_file
from visualize_f import visualize_trajectory, visualize_heatmap, save_figure_to_html

floor_data_dir = './data/site1/F1'
path_data_dir = floor_data_dir + '/path_data_files'
floor_plan_filename = floor_data_dir + '/floor_image.png'
floor_info_filename = floor_data_dir + '/floor_info.json'

save_dir = './output/site1/F1'
path_image_save_dir = save_dir + '/path_images'
step_position_image_save_dir = save_dir
magn_image_save_dir = save_dir
wifi_image_save_dir = save_dir + '/wifi_images'
ibeacon_image_save_dir = save_dir + '/ibeacon_images'
wifi_count_image_save_dir = save_dir


def calibrate_magnetic_wifi_ibeacon_to_position(path_file_list):
    mwi_datas = {}
    for path_filename in path_file_list:
        print(f'Processing {path_filename}...')

        path_datas = read_data_file(path_filename)
        acce_datas = path_datas.acce
        magn_datas = path_datas.magn
        ahrs_datas = path_datas.ahrs
        wifi_datas = path_datas.wifi
        ibeacon_datas = path_datas.ibeacon
        posi_datas = path_datas.waypoint

        step_positions = compute_step_positions(acce_datas, ahrs_datas, posi_datas)
        # visualize_trajectory(posi_datas[:, 1:3], floor_plan_filename, width_meter, height_meter, title='Ground Truth', show=True)
        # visualize_trajectory(step_positions[:, 1:3], floor_plan_filename, width_meter, height_meter, title='Step Position', show=True)

        if wifi_datas.size != 0:
            sep_tss = np.unique(wifi_datas[:, 0].astype(float))
            wifi_datas_list = split_ts_seq(wifi_datas, sep_tss)
            for wifi_ds in wifi_datas_list:
                diff = np.abs(step_positions[:, 0] - float(wifi_ds[0, 0]))
                index = np.argmin(diff)
                target_xy_key = tuple(step_positions[index, 1:3])
                if target_xy_key in mwi_datas:
                    mwi_datas[target_xy_key]['wifi'] = np.append(mwi_datas[target_xy_key]['wifi'], wifi_ds, axis=0)
                else:
                    mwi_datas[target_xy_key] = {
                        'magnetic': np.zeros((0, 4)),
                        'wifi': wifi_ds,
                        'ibeacon': np.zeros((0, 3))
                    }

        if ibeacon_datas.size != 0:
            sep_tss = np.unique(ibeacon_datas[:, 0].astype(float))
            ibeacon_datas_list = split_ts_seq(ibeacon_datas, sep_tss)
            for ibeacon_ds in ibeacon_datas_list:
                diff = np.abs(step_positions[:, 0] - float(ibeacon_ds[0, 0]))
                index = np.argmin(diff)
                target_xy_key = tuple(step_positions[index, 1:3])
                if target_xy_key in mwi_datas:
                    mwi_datas[target_xy_key]['ibeacon'] = np.append(mwi_datas[target_xy_key]['ibeacon'], ibeacon_ds, axis=0)
                else:
                    mwi_datas[target_xy_key] = {
                        'magnetic': np.zeros((0, 4)),
                        'wifi': np.zeros((0, 5)),
                        'ibeacon': ibeacon_ds
                    }

        sep_tss = np.unique(magn_datas[:, 0].astype(float))
        magn_datas_list = split_ts_seq(magn_datas, sep_tss)
        for magn_ds in magn_datas_list:
            diff = np.abs(step_positions[:, 0] - float(magn_ds[0, 0]))
            index = np.argmin(diff)
            target_xy_key = tuple(step_positions[index, 1:3])
            if target_xy_key in mwi_datas:
                mwi_datas[target_xy_key]['magnetic'] = np.append(mwi_datas[target_xy_key]['magnetic'], magn_ds, axis=0)
            else:
                mwi_datas[target_xy_key] = {
                    'magnetic': magn_ds,
                    'wifi': np.zeros((0, 5)),
                    'ibeacon': np.zeros((0, 3))
                }

    return mwi_datas


def extract_magnetic_strength(mwi_datas):
    magnetic_strength = {}
    for position_key in mwi_datas:
        # print(f'Position: {position_key}')

        magnetic_data = mwi_datas[position_key]['magnetic']
        magnetic_s = np.mean(np.sqrt(np.sum(magnetic_data[:, 1:4] ** 2, axis=1)))
        magnetic_strength[position_key] = magnetic_s

    return magnetic_strength


def extract_wifi_rssi(mwi_datas):
    wifi_rssi = {}
    for position_key in mwi_datas:
        # print(f'Position: {position_key}')

        wifi_data = mwi_datas[position_key]['wifi']
        for wifi_d in wifi_data:
            bssid = wifi_d[2]
            rssi = int(wifi_d[3])

            if bssid in wifi_rssi:
                position_rssi = wifi_rssi[bssid]
                if position_key in position_rssi:
                    old_rssi = position_rssi[position_key][0]
                    old_count = position_rssi[position_key][1]
                    position_rssi[position_key][0] = (old_rssi * old_count + rssi) / (old_count + 1)
                    position_rssi[position_key][1] = old_count + 1
                else:
                    position_rssi[position_key] = np.array([rssi, 1])
            else:
                position_rssi = {}
                position_rssi[position_key] = np.array([rssi, 1])

            wifi_rssi[bssid] = position_rssi

    return wifi_rssi


def extract_ibeacon_rssi(mwi_datas):
    ibeacon_rssi = {}
    for position_key in mwi_datas:
        # print(f'Position: {position_key}')

        ibeacon_data = mwi_datas[position_key]['ibeacon']
        for ibeacon_d in ibeacon_data:
            ummid = ibeacon_d[1]
            rssi = int(ibeacon_d[2])

            if ummid in ibeacon_rssi:
                position_rssi = ibeacon_rssi[ummid]
                if position_key in position_rssi:
                    old_rssi = position_rssi[position_key][0]
                    old_count = position_rssi[position_key][1]
                    position_rssi[position_key][0] = (old_rssi * old_count + rssi) / (old_count + 1)
                    position_rssi[position_key][1] = old_count + 1
                else:
                    position_rssi[position_key] = np.array([rssi, 1])
            else:
                position_rssi = {}
                position_rssi[position_key] = np.array([rssi, 1])

            ibeacon_rssi[ummid] = position_rssi

    return ibeacon_rssi


def extract_wifi_count(mwi_datas):
    wifi_counts = {}
    for position_key in mwi_datas:
        # print(f'Position: {position_key}')

        wifi_data = mwi_datas[position_key]['wifi']
        count = np.unique(wifi_data[:, 2]).shape[0]
        wifi_counts[position_key] = count

    return wifi_counts


if __name__ == "__main__":
    Path(path_image_save_dir).mkdir(parents=True, exist_ok=True)
    Path(magn_image_save_dir).mkdir(parents=True, exist_ok=True)
    Path(wifi_image_save_dir).mkdir(parents=True, exist_ok=True)
    Path(ibeacon_image_save_dir).mkdir(parents=True, exist_ok=True)

    with open(floor_info_filename) as f:
        floor_info = json.load(f)
    width_meter = floor_info["map_info"]["width"]
    height_meter = floor_info["map_info"]["height"]

    path_filenames = list(Path(path_data_dir).resolve().glob("*.txt"))

    # 1. visualize ground truth positions
    print('Visualizing ground truth positions...')
    for path_filename in path_filenames:
        print(f'Processing file: {path_filename}...')

        path_data = read_data_file(path_filename)
        path_id = path_filename.name.split(".")[0]
        fig = visualize_trajectory(path_data.waypoint[:, 1:3], floor_plan_filename, width_meter, height_meter, title=path_id, show=False)
        html_filename = f'{path_image_save_dir}/{path_id}.html'
        html_filename = str(Path(html_filename).resolve())
        save_figure_to_html(fig, html_filename)

    # 2. visualize step position, magnetic, wifi, ibeacon
    print('Visualizing more information...')
    mwi_datas = calibrate_magnetic_wifi_ibeacon_to_position(path_filenames)

    step_positions = np.array(list(mwi_datas.keys()))
    fig = visualize_trajectory(step_positions, floor_plan_filename, width_meter, height_meter, mode='markers', title='Step Position', show=True)
    html_filename = f'{step_position_image_save_dir}/step_position.html'
    html_filename = str(Path(html_filename).resolve())
    save_figure_to_html(fig, html_filename)

    magnetic_strength = extract_magnetic_strength(mwi_datas)
    heat_positions = np.array(list(magnetic_strength.keys()))
    heat_values = np.array(list(magnetic_strength.values()))
    fig = visualize_heatmap(heat_positions, heat_values, floor_plan_filename, width_meter, height_meter, colorbar_title='mu tesla', title='Magnetic Strength', show=True)
    html_filename = f'{magn_image_save_dir}/magnetic_strength.html'
    html_filename = str(Path(html_filename).resolve())
    save_figure_to_html(fig, html_filename)

    wifi_rssi = extract_wifi_rssi(mwi_datas)
    print(f'This floor has {len(wifi_rssi.keys())} wifi aps')
    ten_wifi_bssids = list(wifi_rssi.keys())[0:10]
    print('Example 10 wifi ap bssids:\n')
    for bssid in ten_wifi_bssids:
        print(bssid)
    target_wifi = input(f"Please input target wifi ap bssid:\n")
    # target_wifi = '1e:74:9c:a7:b2:e4'
    heat_positions = np.array(list(wifi_rssi[target_wifi].keys()))
    heat_values = np.array(list(wifi_rssi[target_wifi].values()))[:, 0]
    fig = visualize_heatmap(heat_positions, heat_values, floor_plan_filename, width_meter, height_meter, colorbar_title='dBm', title=f'Wifi: {target_wifi} RSSI', show=True)
    html_filename = f'{wifi_image_save_dir}/{target_wifi.replace(":", "-")}.html'
    html_filename = str(Path(html_filename).resolve())
    save_figure_to_html(fig, html_filename)

    ibeacon_rssi = extract_ibeacon_rssi(mwi_datas)
    print(f'This floor has {len(ibeacon_rssi.keys())} ibeacons')
    ten_ibeacon_ummids = list(ibeacon_rssi.keys())[0:10]
    print('Example 10 ibeacon UUID_MajorID_MinorIDs:\n')
    for ummid in ten_ibeacon_ummids:
        print(ummid)
    target_ibeacon = input(f"Please input target ibeacon UUID_MajorID_MinorID:\n")
    # target_ibeacon = 'FDA50693-A4E2-4FB1-AFCF-C6EB07647825_10073_61418'
    heat_positions = np.array(list(ibeacon_rssi[target_ibeacon].keys()))
    heat_values = np.array(list(ibeacon_rssi[target_ibeacon].values()))[:, 0]
    fig = visualize_heatmap(heat_positions, heat_values, floor_plan_filename, width_meter, height_meter, colorbar_title='dBm', title=f'iBeacon: {target_ibeacon} RSSI', show=True)
    html_filename = f'{ibeacon_image_save_dir}/{target_ibeacon}.html'
    html_filename = str(Path(html_filename).resolve())
    save_figure_to_html(fig, html_filename)

    wifi_counts = extract_wifi_count(mwi_datas)
    heat_positions = np.array(list(wifi_counts.keys()))
    heat_values = np.array(list(wifi_counts.values()))
    # filter out positions that no wifi detected
    mask = heat_values != 0
    heat_positions = heat_positions[mask]
    heat_values = heat_values[mask]
    fig = visualize_heatmap(heat_positions, heat_values, floor_plan_filename, width_meter, height_meter, colorbar_title='number', title=f'Wifi Count', show=True)
    html_filename = f'{wifi_count_image_save_dir}/wifi_count.html'
    html_filename = str(Path(html_filename).resolve())
    save_figure_to_html(fig, html_filename)

    print('fff')

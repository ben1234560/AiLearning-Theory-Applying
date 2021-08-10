# Indoor Location Competition 2.0 (Sample Data and Code)

This repository contains sample data and code for [Indoor Location Competition 2.0](https://aka.ms/location20), a continuation of Microsoft Indoor Location Competition. Competition this year will be completely virtual and evaluated on large-scale real indoor location datasets. The dataset to be released consists of dense indoor signatures of WiFi, geomagnetic field, iBeacons etc., as well as ground truth collected from hundreds of buildings in Chinese cities. 

## Webinar Video
We held a webinar in July, the video is [here](https://www.youtube.com/watch?v=xt3OzMC-XMU).

## Sample Data

`data` folder contains indoor traces from two sites. Each trace (`*.txt`) corresponds to an indoor path between position p<sub>1</sub> and p<sub>2</sub> walked by a site-surveyor. During the walk, site-surveyor is holding an Android smartphone flat in front of his body, and a sensor data recording app is running on the device to collect IMU (accelerometer, gyroscope) and geomagnetic field (magnetometer) readings, as well as WiFi and Bluetooth iBeacon scanning results. A detailed description of the format of trace file is shown below. In addition to raw traces, floor plan metadata (e.g., raster image, size, GeoJSON) are also included for each floor. 

### Trace File Format（*.txt）

| Time | Data Type                                           | Value                                  |            |       |           |              |            |            |                                 |
|----------------------|-----------------------------------------------------|------------------------------------------|-------------------|--------------|------------------|---------------------|-------------------|-------------------|----------------------------------------|
| 1574659531598        | TYPE\_WAYPOINT                                      | 196\.41757                               | 117\.84907        |              |                  |                     |                   |                   |                                        |
|                      | Location surveyor labeled on the map       | Coordinate x (meter)                             | Coordiante y (meter)     |              |                  |                     |                   |                   |                                        |
|                      |                                                     |                                          |                   |              |                  |                     |                   |                   |                                        |
| 1574659531695        | TYPE\_ACCELEROMETER                                 | \-1\.7085724                             | \-0\.274765       | 16\.657166   | 2                |                     |                   |                   |                                        |
|                      | Android Sensor\.TYPE\_ACCELEROMETER                 | X axis                                   | Y axis            | Z axis       | accuracy         |                     |                   |                   |                                        |
| 1574659531695        | TYPE\_GYROSCOPE                                     | \-0\.3021698                             | 0\.2773285        | 0\.107543945 | 3                |                     |                   |                   |                                        |
|                      | Android Sensor\.TYPE\_GYROSCOPE                     | X axis                                   | Y axis            | Z axis       | accuracy         |                     |                   |                   |                                        |
| 1574659531695        | TYPE\_MAGNETIC\_FIELD                               | 20\.181274                               | 16\.209412        | \-32\.22046  | 3                |                     |                   |                   |                                        |
|                      | Android Sensor\.TYPE\_MAGNETIC\_FIELD               | X axis                                   | Y axis            | Z axis       | accuracy         |                     |                   |                   |                                        |
| 1574659531695        | TYPE\_ROTATION\_VECTOR                              | \-0\.00855688                            | 0\.051367603      | 0\.362504    | 3                |                     |                   |                   |                                        |
|                      | Android Sensor\.TYPE\_ROTATION\_VECTOR              | X axis                                   | Y axis            | Z axis       | accuracy         |                     |                   |                   |                                        |
|                      |                                                     |                                          |                   |              |                  |                     |                   |                   |                                        |
| 1574659531695        | TYPE\_ACCELEROMETER\_UNCALIBRATED                   | \-1\.7085724                             | \-0\.274765       | 16\.657166   | 0\.0             | 0\.0                | 0\.0              | 3                 |                                        |
|                      | Android Sensor\.TYPE\_ACCELEROMETER\_UNCALIBRATED   | X axis                                   | Y axis            | Z axis       | X axis           | Y axis              | Z axis            | accuracy          |                                        |
| 1574659531695        | TYPE\_GYROSCOPE\_UNCALIBRATED                       | \-0\.42333984                            | 0\.20202637       | 0\.09623718  | \-7\.9345703E\-4 | 3\.2043457E\-4      | 4\.119873E\-4     | 3                 |                                        |
|                      | Android Sensor\.TYPE\_GYROSCOPE\_UNCALIBRATED       | X axis                                   | Y axis            | Z axis       | X axis           | Y axis              | Z axis            | accuracy          |                                        |
| 1574659531695        | TYPE\_MAGNETIC\_FIELD\_UNCALIBRATED                 | \-29\.830933                             | \-26\.36261       | \-300\.3006  | \-50\.012207     | \-42\.57202         | \-268\.08014      | 3                 |                                        |
|                      | Android Sensor\.TYPE\_MAGNETIC\_FIELD\_UNCALIBRATED | X axis                                   | Y axis            | Z axis       | X axis           | Y axis              | Z axis            | accuracy          |                                        |
|                      |                                                     |                                          |                   |              |                  |                     |                   |                   |                                        |
| 1574659533190        | TYPE\_WIFI                                          | intime\_free                             | 0e:74:9c:a7:b2:e4 | \-43         | 5805             | 1574659532305       |                   |                   |                                        |
|                      | Wi\-Fi data                                         | ssid                                     | bssid             | RSSI         | frequency        | last seen timestamp |                   |                   |                                        |
|                      |                                                     |                                          |                   |              |                  |                     |                   |                   |                                        |
| 1574659532751        | TYPE\_BEACON                                        | FDA50693\-A4E2\-4FB1\-AFCF\-C6EB07647825 | 10073             | 61418        | \-65             | \-82                | 5\.50634293288929 | 6B:11:4C:D1:29:F2 | 1574659532751                          |
|                      | iBeacon data                                        | UUID                                     | MajorID           | MinorID      | Tx Power         | RSSI                | Distance          | MAC Address       | same with Unix time, padding data |


The first column is Unix Time in millisecond. In specific, we use SensorEvent.timestamp for sensor data and system time for WiFi and Bluetooth scans. 

The second column is the data type (ten in total).
* TYPE_ACCELEROMETER
* TYPE_MAGNETIC_FIELD
* TYPE_GYROSCOPE
* TYPE_ROTATION_VECTOR
* TYPE_MAGNETIC_FIELD_UNCALIBRATED
* TYPE_GYROSCOPE_UNCALIBRATED
* TYPE_ACCELEROMETER_UNCALIBRATED
* TYPE_WIFI
* TYPE_BEACON
* TYPE_WAYPOINT: ground truth location labeled by the surveyor

Data values start from the third column. 

Column 3-5 of TYPE_ACCELEROMETER、TYPE_MAGNETIC_FIELD、TYPE_GYROSCOPE、TYPE_ROTATION_VECTOR are SensorEvent.values[0-2] from the callback function onSensorChanged(). Column 6 is SensorEvent.accuracy.

Column 3-8 of TYPE_ACCELEROMETER_UNCALIBRATED、TYPE_GYROSCOPE_UNCALIBRATED、TYPE_MAGNETIC_FIELD_UNCALIBRATED are SensorEvent.values[0-5] from the callback function onSensorChanged(). Column 9 is SensorEvent.accuracy.

Values of TYPE_BEACON are obtained from ScanRecord.getBytes(). The results are decoded based on iBeacon protocol using the code below. 
```
val major = ((scanRecord[startByte + 20].toInt() and 0xff) * 0x100 + (scanRecord[startByte + 21].toInt() and 0xff))
val minor = ((scanRecord[startByte + 22].toInt() and 0xff) * 0x100 + (scanRecord[startByte + 23].toInt() and 0xff))
val txPower = scanRecord[startByte + 24]
```
Distance in column 8 is calculated as 
```
private static double calculateDistance(int txPower, double rssi) {
  if (rssi == 0) {
    return -1.0; // if we cannot determine distance, return -1.
  }
  double ratio = rssi*1.0/txPower;
  if (ratio < 1.0) {
    return Math.pow(ratio,10);
  }
  else {
    double accuracy =  (0.89976)*Math.pow(ratio,7.7095) + 0.111;
    return accuracy;
  }
}
```

### References:  
https://developer.android.com/guide/topics/sensors  
https://developer.android.com/reference/android/net/wifi/ScanResult.html  
https://developer.android.com/reference/android/bluetooth/le/ScanRecord



## Sample Code

Along with sample data from two sites, this repo also provides several scripts on parsing and analyzing indoor traces. All scripts are tested with Python 3.6.9 on both Windows 10 and Mac OS 15. 

### How to run the code
`python main.py`

#### Main functions

| Functions                                     | Output                                      |
|-----------------------------------------------|---------------------------------------------|
| Ground truth location visualization           | output/site1/F1/path_images                 |
| Sample step detection and visualization       | output/site1/F1/step_position.html          |
| Geo-magnetic field intensity visualization    | output/site1/F1/magnetic_strength.html      |
| WiFi RSSI heatmap generation                  | output/site1/F1/wifi_images                 |
| iBeacon RSSI heatmap generation               | output/site1/F1/ibeacon_images              |
| WiFi SSID counts visualization                | output/site1/F1/wifi_count.html             |


## Contents
```
indoor-location-competition-20
│   README.md
│   main.py                                                      //main function of the sample code
|   compute_f.py                                                 //data processing functions
|   io_f.py                                                      //data preprocessing functions
|   visualize_f.py                                               //visualization function
│
└───data                                                         //raw data from two sites
      └───site1
      |     └───B1                                               //traces from one floor
      |     |    └───path_data_files                             
      |     |    |          └───5dda14a2c5b77e0006b17533.txt     //trace file
      |     |    |          |   ...
      |     |    |
      |     |    |   floor_image.png                             //raster floor plan
      |     |    |   floor_info.json                             //floor size info
      |     |    |   geojson_map.json                            //floor plan in vector format (GeoJSON)
      |     |
      |     └───F1
      |     │   ...
      |
      └───site2
            │   ...
```


## License

This repository is licensed with the [MIT license](./LICENSE).

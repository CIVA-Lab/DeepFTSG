% % baseline
threshold('baseline/highway/', 470, 1700);
threshold('baseline/office/', 570, 2050);
threshold('baseline/pedestrians/', 300, 1099);
threshold('baseline/PETS2006/', 300, 1200);

% % lowFramerate
threshold('lowFramerate/port_0_17fps/', 1000, 3000);
threshold('lowFramerate/tramCrossroad_1fps/', 400, 900);
threshold('lowFramerate/tunnelExit_0_35fps/', 2000, 4000);
threshold('lowFramerate/turnpike_0_5fps/', 800, 1500);

% % cameraJitter
threshold('cameraJitter/badminton/', 800, 1150);
threshold('cameraJitter/boulevard/', 790, 2500);
threshold('cameraJitter/sidewalk/', 800, 1200);
threshold('cameraJitter/traffic/', 900, 1570);
 
% % nightVideos
threshold('nightVideos/bridgeEntry/', 1000, 2500);
threshold('nightVideos/busyBoulvard/', 730, 2760);
threshold('nightVideos/fluidHighway/', 400, 1364);
threshold('nightVideos/tramStation/', 500, 3000);
threshold('nightVideos/winterStreet/', 900, 1785);
threshold('nightVideos/streetCornerAtNight/', 800, 5200);
  
% % badWeather
threshold('badWeather/skating/', 800, 3900);
threshold('badWeather/wetSnow/', 500, 3500);
threshold('badWeather/snowFall/', 800, 6500);
threshold('badWeather/blizzard/', 900, 7000);
  
% % dynamicBackground
threshold('dynamicBackground/canoe/', 800, 1189);
threshold('dynamicBackground/fall/', 1000, 4000);
threshold('dynamicBackground/fountain01/', 400, 1184);
threshold('dynamicBackground/fountain02/', 500, 1499);
threshold('dynamicBackground/overpass/', 1000, 3000);
threshold('dynamicBackground/boats/', 1900, 7999);
  
% % intermittentObjectMotion
threshold('intermittentObjectMotion/abandonedBox/', 2450, 4500);
threshold('intermittentObjectMotion/parking/', 1100, 2500);
threshold('intermittentObjectMotion/sofa/', 500, 2750);
threshold('intermittentObjectMotion/streetLight/', 175, 3200);
threshold('intermittentObjectMotion/tramstop/', 1320, 3200);
threshold('intermittentObjectMotion/winterDriveway/', 1000, 2500);
  
% % PTZ
threshold('PTZ/continuousPan/', 600, 1700);
threshold('PTZ/intermittentPan/', 1200, 3500);
threshold('PTZ/twoPositionPTZCam/', 800, 2300);
threshold('PTZ/zoomInZoomOut/', 500, 1130);
 
% % shadow
threshold('shadow/backdoor/', 400, 2000);
threshold('shadow/bungalows/', 300, 1700);
threshold('shadow/busStation/', 300, 1250);
threshold('shadow/copyMachine/', 500, 3400);
threshold('shadow/peopleInShade/', 250, 1199);
threshold('shadow/cubicle/', 1100, 7400);
  
% % thermal
threshold('thermal/diningRoom/', 700, 3700);
threshold('thermal/library/', 600, 4900);
threshold('thermal/park/', 250, 600);
threshold('thermal/corridor/', 500, 5400);
threshold('thermal/lakeSide/', 1000, 6500);
  
% % turbulence
threshold('turbulence/turbulence0/', 1000, 5000);
threshold('turbulence/turbulence1/', 1200, 4000);
threshold('turbulence/turbulence2/', 500, 4500);
threshold('turbulence/turbulence3/', 800, 2200);
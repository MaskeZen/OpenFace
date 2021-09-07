# Notas

## Build

`cmake --build ~/repo/OpenFace/build --config Debug --target FacePreDetectionImg -j 6 --`

## Run

Ejemplos

**FacePreDetectionImg**
`./build/bin/FacePreDetectionImg -fdir /home/maske/repo/work/ituy/mediapipe/img/faces/set04`

**FeatureExtraction**
`./build/bin/FeatureExtraction -verbose -pose -device /dev/video1`

## Enlaces

- [camera-rotation-in-world-coordinates](https://blender.stackexchange.com/questions/134770/camera-rotation-in-world-coordinates)
- [What is the world coordinate? #89](https://github.com/TadasBaltrusaitis/OpenFace/issues/89)

### Instalación

- [opencv instalación](https://docs.opencv.org/master/d7/d9f/tutorial_linux_install.html)

### Daemon

- [Ejemplo de daemon 1](https://github.com/jirihnidek/daemon)
- [Ejemplo de daemon 2](https://gist.github.com/alexdlaird/3100f8c7c96871c5b94e)
- [Ejemplo de daemon 3](https://gist.github.com/faberyx/b07d146e11efbad1643f3e8ba6f1a475)

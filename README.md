YOLO DL4J
--

This is an UI for the YOLO DL4J Zoo model. The same JavaFX application could be used with other DL4J YOLO models,
hence why it is in an utility directory and not as a sample application.

### yolo
运行YOLOApp.java
自动下载yolo模型 C:\Users\小松\.deeplearning4j\models\yolo2\yolo2_dl4j_inference.v3.zip

### yolo1
修改模型的位置 Yolo2Prediction.java
File locationToSave = new File(....)
运行Yolo2Test

### skil  yolo2模型部署skil
修改YoloClientTest.java
```bash
defaultClient.setBasePath("http://192.168.1.128:9008");
loginRequest.setUserId("admin");
loginRequest.setPassword("123456");
```
运行YoloClientTest.java


修改ClientSkil.java
运行ClientSkil.java
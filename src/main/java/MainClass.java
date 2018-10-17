import org.bytedeco.javacpp.opencv_core;

import javax.swing.*;
import java.awt.*;

import static org.bytedeco.javacpp.opencv_core.CV_8UC1;
import static org.bytedeco.javacpp.opencv_core.merge;
import static org.bytedeco.javacpp.opencv_core.split;
import static org.bytedeco.javacpp.opencv_imgcodecs.IMREAD_COLOR;
import static org.bytedeco.javacpp.opencv_imgcodecs.imread;
import static org.bytedeco.javacpp.opencv_imgproc.resize;

public class MainClass {

    public static void main(String[] args) {
        String baseFile = "data/tower.jpg";
        opencv_core.Mat inputMat = imread("data/tower.jpg", IMREAD_COLOR);
        resize(inputMat, inputMat, new opencv_core.Size(800, 600));

        JFrame frame = new MainForm(inputMat);
        frame.setSize(new Dimension(800, 600));
        frame.setVisible(true);
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
    }
}

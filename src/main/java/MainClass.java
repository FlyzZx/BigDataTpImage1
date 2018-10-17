import javax.swing.*;
import java.awt.*;

import static org.bytedeco.javacpp.opencv_core.Mat;
import static org.bytedeco.javacpp.opencv_core.Size;
import static org.bytedeco.javacpp.opencv_imgcodecs.IMREAD_COLOR;
import static org.bytedeco.javacpp.opencv_imgcodecs.imread;
import static org.bytedeco.javacpp.opencv_imgproc.resize;

public class MainClass {

    public static void main(String[] args) {
        Mat image = imread("data/tower.jpg", IMREAD_COLOR);
        resize(image, image, new Size(800, 600));

        JFrame frame = new MainForm();
        frame.setSize(new Dimension(800, 600));
        frame.setVisible(true);
    }

}

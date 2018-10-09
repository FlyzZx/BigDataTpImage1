import org.bytedeco.javacpp.indexer.UByteIndexer;
import org.bytedeco.javacpp.indexer.UByteRawIndexer;
import org.bytedeco.javacpp.opencv_core;
import org.bytedeco.javacv.CanvasFrame;
import org.bytedeco.javacv.OpenCVFrameConverter;

import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.util.ArrayList;

import static org.bytedeco.javacpp.opencv_core.*;

import static org.bytedeco.javacpp.opencv_imgcodecs.*;

import static org.bytedeco.javacpp.opencv_imgproc.*;

public class MainClass {

    public static void main(String[] args) {
        Mat image = imread("data/tower.jpg", IMREAD_COLOR);
        resize(image, image, new Size(800, 600));

        JFrame frame = new MainForm();
        frame.setSize(new Dimension(800, 600));
        frame.setVisible(true);
    }

}

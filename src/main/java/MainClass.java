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

    private static void splitRGBShow(opencv_core.Mat image, boolean R, boolean G, boolean B){
        opencv_core.MatVector rgbSplit = new opencv_core.MatVector();
        opencv_core.MatVector choosenSplit = new opencv_core.MatVector();
        opencv_core.Mat red = new opencv_core.Mat(image.rows(),image.cols(),CV_8UC1);
        opencv_core.Mat green = new opencv_core.Mat(image.rows(),image.cols(),CV_8UC1);
        opencv_core.Mat blue = new opencv_core.Mat(image.rows(),image.cols(),CV_8UC1);
        opencv_core.Mat result = new opencv_core.Mat();
        String windowName = "";
        split(image, rgbSplit);

        if(R){red = rgbSplit.get(2);windowName += "R";}
        if(G){green = rgbSplit.get(1);windowName += "G";}
        if(B){blue = rgbSplit.get(0);windowName += "B";}

        choosenSplit.push_back(blue);
        choosenSplit.push_back(green);
        choosenSplit.push_back(red);
        merge(choosenSplit,result);
        String name = windowName.replaceAll("(?<=.)(?=.)","+");
        Show(result, name);

        /*
        Mat cercle = imread("data/Cercle.png",IMREAD_COLOR);
        splitRGBShow(cercle,true,false,false);
        splitRGBShow(cercle,false,true,false);
        splitRGBShow(cercle,false,false,true);
        splitRGBShow(cercle,true,true,false);
        splitRGBShow(cercle,false,true,true);
        splitRGBShow(cercle,true,false,true);
        splitRGBShow(cercle,true,true,true);
         */

    }
}

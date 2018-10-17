import org.bytedeco.javacpp.opencv_core;

import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

import static org.bytedeco.javacpp.opencv_imgcodecs.*;
import static org.bytedeco.javacpp.opencv_imgproc.resize;

public class MainForm extends JFrame {

    private JButton lectureEtAffichageDButton;
    private JButton opérationMorphologiquesEtSeuillagesButton;
    private JButton canauxDeCouleursButton;
    private JButton calculDHistogrammeButton;
    private JButton lookUpTableButton;
    private JButton segmentationButton;
    private JPanel jPanel;
    private JButton siftMatching;
    private JButton compare;
    private JButton rgbSplit;
    private JButton detectFace;
    private JButton trainFace;
    private JButton DetectionAndReconnaissance;

    public MainForm(opencv_core.Mat inputMat) throws HeadlessException {
        this.add(jPanel);

        lectureEtAffichageDButton.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e) {
                BigMData.Show(inputMat, "Affichage d'une image");
            }
        });

        opérationMorphologiquesEtSeuillagesButton.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e) {
                BigMData.morpho(inputMat);
            }
        });

        canauxDeCouleursButton.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e) {
                BigMData.wreckedtomestleseulRGB(inputMat);
            }
        });

        calculDHistogrammeButton.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e) {
                BigMData.histogramme(inputMat);
            }
        });

        lookUpTableButton.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e) {
                BigMData.withLut(inputMat);
            }
        });

        segmentationButton.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e) {
                BigMData.segmentationKmeans(inputMat);
            }
        });

        siftMatching.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e) {
                BigMData.siftMatching(inputMat);
            }
        });

        compare.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e) {
                BigMData.siftTraining(inputMat);
            }
        });

        rgbSplit.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e) {
                opencv_core.Mat cercle = imread("data/Cercle.png",IMREAD_COLOR);
                BigMData.splitRGBShow(cercle,true,false,false);
                BigMData.splitRGBShow(cercle,false,true,false);
                BigMData.splitRGBShow(cercle,false,false,true);
                BigMData.splitRGBShow(cercle,true,true,false);
                BigMData.splitRGBShow(cercle,false,true,true);
                BigMData.splitRGBShow(cercle,true,false,true);
                BigMData.splitRGBShow(cercle,true,true,true);
            }
        });

        detectFace.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e) {
                opencv_core.Mat face = imread("data/face.jpg",IMREAD_COLOR);
                BigMData.detectFace(face);
            }
        });

        trainFace.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e) {
                opencv_core.Mat face = imread("data/bill.jpg", CV_LOAD_IMAGE_GRAYSCALE);
                BigMData.trainFace();//Training
                int classe = BigMData.prediction(face);
                System.out.println("prediction:" + classe);
            }
        });

        DetectionAndReconnaissance.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                BigMData.trainFace();
                BigMData.cropImage("data/bill.jpg");
            }
        });
    }
}

import org.bytedeco.javacpp.opencv_core;
import org.opencv.core.Mat;

import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

import static org.bytedeco.javacpp.opencv_imgcodecs.CV_LOAD_IMAGE_COLOR;
import static org.bytedeco.javacpp.opencv_imgcodecs.imread;

public class MainForm extends JFrame {

    private JButton lectureEtAffichageDButton;
    private JButton opérationMorphologiquesEtSeuillagesButton;
    private JButton canauxDeCouleursButton;
    private JButton calculDHistogrammeButton;
    private JButton lookUpTableButton;
    private JButton segmentationButton;
    private JPanel jPanel;

    public MainForm() throws HeadlessException {
        String baseFile = "data/tower.jpg";
        final opencv_core.Mat inputMat = imread(baseFile, CV_LOAD_IMAGE_COLOR);
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
    }
}

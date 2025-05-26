package com.celanim.cartoonify;

import javax.swing.*;
import java.awt.*;
import java.awt.event.*;
import java.awt.image.BufferedImage;
import javax.imageio.ImageIO;
import java.io.File;
import java.io.IOException;

public class CartoonifyGUI extends JFrame {

    // Instance Variables (Components)
    private JButton uploadButton;
    private JButton cartoonifyButton;
    private JButton saveButton;
    private JLabel originalImageLabel;
    private JLabel cartoonifiedImageLabel;
    private JPanel imagePanel; // Panel to hold the two image labels
    private JFileChooser fileChooser;

    // Image data
    private BufferedImage originalImage;
    private BufferedImage cartoonImage;

    // Cartoonify instance
    private Cartoonify cartoonifier;

    public CartoonifyGUI() {
        // Initialize Cartoonify
        cartoonifier = new Cartoonify();

        // Set up JFrame properties
        setTitle("Cartoonify Image");
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setSize(800, 600);
        setLayout(new BorderLayout()); // Main layout for the JFrame

        // Initialize UI components
        uploadButton = new JButton("Upload Image");
        cartoonifyButton = new JButton("Cartoonify Image");
        saveButton = new JButton("Save Cartoonified Image");

        originalImageLabel = new JLabel();
        originalImageLabel.setHorizontalAlignment(JLabel.CENTER); // Center align images/text
        cartoonifiedImageLabel = new JLabel();
        cartoonifiedImageLabel.setHorizontalAlignment(JLabel.CENTER); // Center align images/text

        fileChooser = new JFileChooser();

        // Set initial states for buttons
        cartoonifyButton.setEnabled(false);
        saveButton.setEnabled(false);

        // Layout components

        // Button Panel
        JPanel buttonPanel = new JPanel(new FlowLayout());
        buttonPanel.add(uploadButton);
        buttonPanel.add(cartoonifyButton);
        buttonPanel.add(saveButton);

        // Image Panel
        imagePanel = new JPanel(new GridLayout(1, 2, 5, 5)); // 1 row, 2 columns, with spacing
        imagePanel.add(new JScrollPane(originalImageLabel)); // Add scroll panes for potentially large images
        imagePanel.add(new JScrollPane(cartoonifiedImageLabel));

        // Add panels to the frame
        add(buttonPanel, BorderLayout.NORTH);
        add(imagePanel, BorderLayout.CENTER);

        // Make the frame visible
        // setVisible(true); // Will be called from main method later
    }

    // ActionListeners and other logic will be added in subsequent steps.

    // Helper method to scale images
    private ImageIcon scaleImage(BufferedImage image, int targetWidth) {
        if (image == null) {
            return null;
        }

        int originalWidth = image.getWidth();
        int originalHeight = image.getHeight();

        if (originalWidth <= 0 || originalHeight <= 0) {
            return new ImageIcon(image); // Should not happen with valid images
        }

        // Calculate new height to maintain aspect ratio
        int newWidth = targetWidth;
        int newHeight = (int) Math.round((double) originalHeight * targetWidth / originalWidth);

        if (newHeight <= 0) { // Ensure height is positive
            newHeight = 1;
        }
        
        Image scaledImage = image.getScaledInstance(newWidth, newHeight, Image.SCALE_SMOOTH);
        return new ImageIcon(scaledImage);
    }

    private void addActionListeners() {
        uploadButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                if (fileChooser.showOpenDialog(CartoonifyGUI.this) == JFileChooser.APPROVE_OPTION) {
                    File selectedFile = fileChooser.getSelectedFile();
                    try {
                        originalImage = ImageIO.read(selectedFile);
                        if (originalImage != null) {
                            // Scale image to fit label - using a fixed width for now
                            // Assuming imagePanel width is roughly 780 (800 - margins), each label can be ~390.
                            // Let's use 380 as target width for scaling.
                            originalImageLabel.setIcon(scaleImage(originalImage, 380));
                            cartoonifyButton.setEnabled(true);
                            cartoonifiedImageLabel.setIcon(null); // Clear previous cartoon
                            cartoonImage = null;
                            saveButton.setEnabled(false);
                        } else {
                            JOptionPane.showMessageDialog(CartoonifyGUI.this,
                                    "Could not read image: Invalid image file.",
                                    "Image Error", JOptionPane.ERROR_MESSAGE);
                            originalImage = null; // Ensure it's null
                            originalImageLabel.setIcon(null);
                            cartoonifyButton.setEnabled(false);
                            saveButton.setEnabled(false);
                        }
                    } catch (IOException ex) {
                        JOptionPane.showMessageDialog(CartoonifyGUI.this,
                                "Error reading image file: " + ex.getMessage(),
                                "File Read Error", JOptionPane.ERROR_MESSAGE);
                        originalImage = null;
                        originalImageLabel.setIcon(null);
                        cartoonifyButton.setEnabled(false);
                        saveButton.setEnabled(false);
                    }
                }
            }
        });

        cartoonifyButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                if (originalImage != null) {
                    try {
                        // Ensure cartoonifier has the latest settings if they were made configurable
                        // For now, default settings are used as per Cartoonify class
                        cartoonImage = cartoonifier.processBufferedImage(originalImage);
                        if (cartoonImage != null) {
                            cartoonifiedImageLabel.setIcon(scaleImage(cartoonImage, 380));
                            saveButton.setEnabled(true);
                        } else {
                             JOptionPane.showMessageDialog(CartoonifyGUI.this,
                                    "Cartoonification resulted in a null image.",
                                    "Processing Error", JOptionPane.ERROR_MESSAGE);
                            saveButton.setEnabled(false);
                        }
                    } catch (IOException ex) {
                        JOptionPane.showMessageDialog(CartoonifyGUI.this,
                                "Error processing image: " + ex.getMessage(),
                                "Processing Error", JOptionPane.ERROR_MESSAGE);
                        cartoonImage = null;
                        cartoonifiedImageLabel.setIcon(null);
                        saveButton.setEnabled(false);
                    } catch (Exception ex) {
                        // Catch any other unexpected errors from processing
                        JOptionPane.showMessageDialog(CartoonifyGUI.this,
                                "An unexpected error occurred during cartoonification: " + ex.getMessage(),
                                "Processing Error", JOptionPane.ERROR_MESSAGE);
                        ex.printStackTrace(); // For developer debugging
                        cartoonImage = null;
                        cartoonifiedImageLabel.setIcon(null);
                        saveButton.setEnabled(false);
                    }
                } else {
                    JOptionPane.showMessageDialog(CartoonifyGUI.this,
                            "Please upload an image first.",
                            "No Image", JOptionPane.WARNING_MESSAGE);
                }
            }
        });

        saveButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                if (cartoonImage != null) {
                    if (fileChooser.showSaveDialog(CartoonifyGUI.this) == JFileChooser.APPROVE_OPTION) {
                        File outputFile = fileChooser.getSelectedFile();
                        String fileName = outputFile.getName();
                        String extension = "png"; // Default extension

                        int dotIndex = fileName.lastIndexOf('.');
                        if (dotIndex > 0 && dotIndex < fileName.length() - 1) {
                            extension = fileName.substring(dotIndex + 1).toLowerCase();
                            if (!extension.equals("png") && !extension.equals("jpg") && !extension.equals("jpeg")) {
                                JOptionPane.showMessageDialog(CartoonifyGUI.this,
                                        "Unsupported file type. Please use .png or .jpg.",
                                        "Save Error", JOptionPane.ERROR_MESSAGE);
                                return; // Early exit if unsupported extension explicitly provided
                            }
                        } else {
                            // No extension or extension is at the beginning/end, append default
                            outputFile = new File(outputFile.getParentFile(), fileName + "." + extension);
                        }
                        
                        // Re-check extension from the potentially modified outputFile name for ImageIO.write
                        String finalFileName = outputFile.getName();
                        int finalDotIndex = finalFileName.lastIndexOf('.');
                        if (finalDotIndex > 0) {
                            extension = finalFileName.substring(finalDotIndex + 1).toLowerCase();
                        }


                        try {
                            boolean success = ImageIO.write(cartoonImage, extension, outputFile);
                            if (success) {
                                JOptionPane.showMessageDialog(CartoonifyGUI.this,
                                        "Image saved successfully to " + outputFile.getAbsolutePath(),
                                        "Save Successful", JOptionPane.INFORMATION_MESSAGE);
                            } else {
                                JOptionPane.showMessageDialog(CartoonifyGUI.this,
                                        "Could not save image. Writer not found for format: " + extension,
                                        "Save Error", JOptionPane.ERROR_MESSAGE);
                            }
                        } catch (IOException ex) {
                            JOptionPane.showMessageDialog(CartoonifyGUI.this,
                                    "Error saving image: " + ex.getMessage(),
                                    "Save Error", JOptionPane.ERROR_MESSAGE);
                        }
                    }
                } else {
                    JOptionPane.showMessageDialog(CartoonifyGUI.this,
                            "No cartoonified image to save.",
                            "No Image", JOptionPane.WARNING_MESSAGE);
                }
            }
        });
    }
    
    // Constructor calls addActionListeners
    public CartoonifyGUI() {
        // Initialize Cartoonify
        cartoonifier = new Cartoonify();

        // Set up JFrame properties
        setTitle("Cartoonify Image");
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setSize(800, 600);
        setLayout(new BorderLayout()); // Main layout for the JFrame

        // Initialize UI components
        uploadButton = new JButton("Upload Image");
        cartoonifyButton = new JButton("Cartoonify Image");
        saveButton = new JButton("Save Cartoonified Image");

        originalImageLabel = new JLabel();
        originalImageLabel.setHorizontalAlignment(JLabel.CENTER); // Center align images/text
        cartoonifiedImageLabel = new JLabel();
        cartoonifiedImageLabel.setHorizontalAlignment(JLabel.CENTER); // Center align images/text

        fileChooser = new JFileChooser();

        // Set initial states for buttons
        cartoonifyButton.setEnabled(false);
        saveButton.setEnabled(false);

        // Layout components

        // Button Panel
        JPanel buttonPanel = new JPanel(new FlowLayout());
        buttonPanel.add(uploadButton);
        buttonPanel.add(cartoonifyButton);
        buttonPanel.add(saveButton);

        // Image Panel
        imagePanel = new JPanel(new GridLayout(1, 2, 5, 5)); // 1 row, 2 columns, with spacing
        imagePanel.add(new JScrollPane(originalImageLabel)); // Add scroll panes for potentially large images
        imagePanel.add(new JScrollPane(cartoonifiedImageLabel));

        // Add panels to the frame
        add(buttonPanel, BorderLayout.NORTH);
        add(imagePanel, BorderLayout.CENTER);

        // Add action listeners
        addActionListeners();

        // Make the frame visible
        // setVisible(true); // Will be called from main method later
    }

    public static void main(String[] args) {
        // Ensure GUI creation and updates are on the Event Dispatch Thread
        SwingUtilities.invokeLater(new Runnable() {
            @Override
            public void run() {
                CartoonifyGUI gui = new CartoonifyGUI();
                gui.setVisible(true);
            }
        });
    }
}

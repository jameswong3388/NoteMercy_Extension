"use client";

import {FilePond, registerPlugin} from "react-filepond";
import "filepond/dist/filepond.min.css";
import FilePondPluginImageExifOrientation from "filepond-plugin-image-exif-orientation";
import FilePondPluginImagePreview from "filepond-plugin-image-preview";
import "filepond-plugin-image-preview/dist/filepond-plugin-image-preview.css";
import {ResizableHandle, ResizablePanel, ResizablePanelGroup} from "@/components/ui/resizable";
import {ScrollArea} from "@/components/ui/scroll-area";
import {useState} from "react";

interface HandwritingStyle {
    score: number;
}

interface HandwritingFeatures {
    handwriting: {
        block_lettering: HandwritingStyle;
        cursive: HandwritingStyle;
        calligraphic: HandwritingStyle;
        italic: HandwritingStyle;
        shorthand: HandwritingStyle;
        print: HandwritingStyle;
    };
    angularity: any;
    uppercase_ratio: any;
    pen_pressure: any;
    vertical_stroke_proportion: any;
    slant_angle: any;
    inter_letter_spacing: any;
    stroke_connectivity: any;
    enclosed_loop_ratio: any;
    curvature_continuity: any;
    stroke_width_variation: any;
    flourish_extension: any;
    artistic_consistency: any;
    stroke_continuity: any;
    smooth_curves: any;
    vertical_alignment: any;
    letter_size_uniformity: any;
    discrete_letter: any;
    processed_image: string;
}

// Register FilePond plugins
registerPlugin(FilePondPluginImageExifOrientation, FilePondPluginImagePreview);

export default function Home() {
    const [originalImage, setOriginalImage] = useState<string | null>(null);
    const [processedImage, setProcessedImage] = useState<string | null>(null);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [features, setFeatures] = useState<HandwritingFeatures | null>(null);

    const handleFileUpload = async (file: Blob) => {
        try {
            setIsLoading(true);
            setError(null);
            
            // Convert file to base64
            const reader = new FileReader();
            reader.onload = async (e) => {
                const base64Image = e.target?.result as string;
                setOriginalImage(base64Image);

                // Remove data URL prefix if present
                const base64Data = base64Image.split(',')[1] || base64Image;

                // Call API
                const response = await fetch('http://localhost:8000/api/v1/extract', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ image: base64Data }),
                });

                if (!response.ok) {
                    const errorData = await response.json().catch(() => ({}));
                    throw new Error(errorData.detail || 'Failed to process image');
                }

                const data = await response.json();
                setProcessedImage(data.processed_image);
                setFeatures(data);
            };
            reader.readAsDataURL(file);
        } catch (error) {
            console.error('Error processing image:', error);
            setError(error instanceof Error ? error.message : 'An error occurred while processing the image');
            setProcessedImage(null);
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div className="h-screen p-4">
            <ResizablePanelGroup direction="horizontal" className="h-full rounded-lg border">
                <ResizablePanel defaultSize={30} minSize={30}>
                    <ResizablePanelGroup direction="vertical" className="h-full">
                        <ResizablePanel defaultSize={20} minSize={20}>
                            <ScrollArea className="h-full p-4">
                                <h2 className="text-lg font-bold mb-4">Upload Handwriting</h2>
                                <FilePond
                                    allowMultiple={false}
                                    acceptedFileTypes={["image/*"]}
                                    labelIdle='Drag & Drop your handwriting image or <span class="filepond--label-action">Browse</span>'
                                    stylePanelLayout="compact"
                                    imagePreviewHeight={100}
                                    credits={false}
                                    onaddfile={(error, file) => {
                                        if (!error) {
                                            handleFileUpload(file.file);
                                        }
                                    }}
                                />
                            </ScrollArea>
                        </ResizablePanel>
                        <ResizableHandle withHandle />
                        <ResizablePanel defaultSize={40} minSize={20}>
                            <ScrollArea className="h-full p-4">
                                <h2 className="text-lg font-bold mb-4">Original Image</h2>
                                <div className="flex items-center justify-center h-[calc(100%-2rem)]">
                                    {originalImage ? (
                                        <img src={originalImage} alt="Original" className="max-w-full max-h-full" />
                                    ) : (
                                        <span className="text-muted-foreground">Original image will appear here</span>
                                    )}
                                </div>
                            </ScrollArea>
                        </ResizablePanel>
                        <ResizableHandle withHandle />
                        <ResizablePanel defaultSize={40} minSize={20}>
                            <ScrollArea className="h-full p-4">
                                <h2 className="text-lg font-bold mb-4">Pre-processed Image</h2>
                                <div className="flex items-center justify-center h-[calc(100%-2rem)]">
                                    {isLoading ? (
                                        <span className="text-muted-foreground">Processing image...</span>
                                    ) : error ? (
                                        <span className="text-red-500">{error}</span>
                                    ) : processedImage ? (
                                        <img src={`data:image/jpeg;base64,${processedImage}`} alt="Processed" className="max-w-full max-h-full" />
                                    ) : (
                                        <span className="text-muted-foreground">Pre-processed image will appear here</span>
                                    )}
                                </div>
                            </ScrollArea>
                        </ResizablePanel>
                    </ResizablePanelGroup>
                </ResizablePanel>
                <ResizableHandle withHandle/>
                <ResizablePanel defaultSize={70} minSize={30}>
                    <ResizablePanelGroup direction="vertical" className="h-full">
                        <ResizablePanel defaultSize={50} minSize={30}>
                            <ScrollArea className="h-full p-4">
                                <h2 className="text-lg font-bold mb-4">Extracted Features</h2>
                                <div className="space-y-4">
                                    {features ? (
                                        <div className="space-y-6">
                                            {/* Block Lettering Features */}
                                            <div>
                                                <h3 className="font-semibold mb-2">Block Lettering Features</h3>
                                                <div className="space-y-2">
                                                    <div className="bg-muted p-3 rounded">
                                                        <h4 className="font-medium">Angularity</h4>
                                                        <pre className="text-sm mt-1 overflow-auto">
                                                            {JSON.stringify(features.angularity, null, 2)}
                                                        </pre>
                                                    </div>
                                                    <div className="bg-muted p-3 rounded">
                                                        <h4 className="font-medium">Uppercase Ratio</h4>
                                                        <pre className="text-sm mt-1 overflow-auto">
                                                            {JSON.stringify(features.uppercase_ratio, null, 2)}
                                                        </pre>
                                                    </div>
                                                    <div className="bg-muted p-3 rounded">
                                                        <h4 className="font-medium">Pen Pressure</h4>
                                                        <pre className="text-sm mt-1 overflow-auto">
                                                            {JSON.stringify(features.pen_pressure, null, 2)}
                                                        </pre>
                                                    </div>
                                                </div>
                                            </div>

                                            {/* Italic Features */}
                                            <div>
                                                <h3 className="font-semibold mb-2">Italic Features</h3>
                                                <div className="space-y-2">
                                                    <div className="bg-muted p-3 rounded">
                                                        <h4 className="font-medium">Vertical Stroke Proportion</h4>
                                                        <pre className="text-sm mt-1 overflow-auto">
                                                            {JSON.stringify(features.vertical_stroke_proportion, null, 2)}
                                                        </pre>
                                                    </div>
                                                    <div className="bg-muted p-3 rounded">
                                                        <h4 className="font-medium">Slant Angle</h4>
                                                        <pre className="text-sm mt-1 overflow-auto">
                                                            {JSON.stringify(features.slant_angle, null, 2)}
                                                        </pre>
                                                    </div>
                                                    <div className="bg-muted p-3 rounded">
                                                        <h4 className="font-medium">Inter-Letter Spacing</h4>
                                                        <pre className="text-sm mt-1 overflow-auto">
                                                            {JSON.stringify(features.inter_letter_spacing, null, 2)}
                                                        </pre>
                                                    </div>
                                                </div>
                                            </div>

                                            {/* Cursive Features */}
                                            <div>
                                                <h3 className="font-semibold mb-2">Cursive Features</h3>
                                                <div className="space-y-2">
                                                    <div className="bg-muted p-3 rounded">
                                                        <h4 className="font-medium">Stroke Connectivity</h4>
                                                        <pre className="text-sm mt-1 overflow-auto">
                                                            {JSON.stringify(features.stroke_connectivity, null, 2)}
                                                        </pre>
                                                    </div>
                                                    <div className="bg-muted p-3 rounded">
                                                        <h4 className="font-medium">Enclosed Loop Ratio</h4>
                                                        <pre className="text-sm mt-1 overflow-auto">
                                                            {JSON.stringify(features.enclosed_loop_ratio, null, 2)}
                                                        </pre>
                                                    </div>
                                                    <div className="bg-muted p-3 rounded">
                                                        <h4 className="font-medium">Curvature Continuity</h4>
                                                        <pre className="text-sm mt-1 overflow-auto">
                                                            {JSON.stringify(features.curvature_continuity, null, 2)}
                                                        </pre>
                                                    </div>
                                                </div>
                                            </div>

                                            {/* Calligraphic Features */}
                                            <div>
                                                <h3 className="font-semibold mb-2">Calligraphic Features</h3>
                                                <div className="space-y-2">
                                                    <div className="bg-muted p-3 rounded">
                                                        <h4 className="font-medium">Stroke Width Variation</h4>
                                                        <pre className="text-sm mt-1 overflow-auto">
                                                            {JSON.stringify(features.stroke_width_variation, null, 2)}
                                                        </pre>
                                                    </div>
                                                    <div className="bg-muted p-3 rounded">
                                                        <h4 className="font-medium">Flourish Extension</h4>
                                                        <pre className="text-sm mt-1 overflow-auto">
                                                            {JSON.stringify(features.flourish_extension, null, 2)}
                                                        </pre>
                                                    </div>
                                                    <div className="bg-muted p-3 rounded">
                                                        <h4 className="font-medium">Artistic Consistency</h4>
                                                        <pre className="text-sm mt-1 overflow-auto">
                                                            {JSON.stringify(features.artistic_consistency, null, 2)}
                                                        </pre>
                                                    </div>
                                                </div>
                                            </div>

                                            {/* Shorthand Features */}
                                            <div>
                                                <h3 className="font-semibold mb-2">Shorthand Features</h3>
                                                <div className="space-y-2">
                                                    <div className="bg-muted p-3 rounded">
                                                        <h4 className="font-medium">Stroke Continuity</h4>
                                                        <pre className="text-sm mt-1 overflow-auto">
                                                            {JSON.stringify(features.stroke_continuity, null, 2)}
                                                        </pre>
                                                    </div>
                                                    <div className="bg-muted p-3 rounded">
                                                        <h4 className="font-medium">Smooth Curves</h4>
                                                        <pre className="text-sm mt-1 overflow-auto">
                                                            {JSON.stringify(features.smooth_curves, null, 2)}
                                                        </pre>
                                                    </div>
                                                </div>
                                            </div>

                                            {/* Print Features */}
                                            <div>
                                                <h3 className="font-semibold mb-2">Print Features</h3>
                                                <div className="space-y-2">
                                                    <div className="bg-muted p-3 rounded">
                                                        <h4 className="font-medium">Vertical Alignment</h4>
                                                        <pre className="text-sm mt-1 overflow-auto">
                                                            {JSON.stringify(features.vertical_alignment, null, 2)}
                                                        </pre>
                                                    </div>
                                                    <div className="bg-muted p-3 rounded">
                                                        <h4 className="font-medium">Letter Size Uniformity</h4>
                                                        <pre className="text-sm mt-1 overflow-auto">
                                                            {JSON.stringify(features.letter_size_uniformity, null, 2)}
                                                        </pre>
                                                    </div>
                                                    <div className="bg-muted p-3 rounded">
                                                        <h4 className="font-medium">Discrete Letter</h4>
                                                        <pre className="text-sm mt-1 overflow-auto">
                                                            {JSON.stringify(features.discrete_letter, null, 2)}
                                                        </pre>
                                                    </div>
                                                </div>
                                            </div>
                                        </div>
                                    ) : (
                                        <div className="flex items-center justify-center h-[calc(100%-8rem)] text-muted-foreground">
                                            Upload an image to see handwriting analysis results
                                        </div>
                                    )}
                                </div>
                            </ScrollArea>
                        </ResizablePanel>
                        <ResizableHandle withHandle />
                        <ResizablePanel defaultSize={50} minSize={30}>
                            <ScrollArea className="h-full p-4">
                                <h2 className="text-lg font-bold mb-4">Handwriting Recognition</h2>
                                {features ? (
                                    <div className="space-y-6">
                                        {/* Handwriting Style Scores */}
                                        <div className="bg-muted p-4 rounded-lg">
                                            <h3 className="font-semibold mb-2">Handwriting Style Scores</h3>
                                            <div className="grid grid-cols-2 gap-2">
                                                {Object.entries(features.handwriting).map(([style, data]) => (
                                                    <div key={style} className="flex justify-between">
                                                        <span className="capitalize">{style.replace('_', ' ')}:</span>
                                                        <span className="font-medium">{(data.score * 100).toFixed(1)}%</span>
                                                    </div>
                                                ))}
                                            </div>
                                        </div>
                                    </div>
                                ) : (
                                    <div className="flex items-center justify-center h-[calc(100%-8rem)] text-muted-foreground">
                                        Upload an image to see recognized text and handwriting style analysis
                                    </div>
                                )}
                            </ScrollArea>
                        </ResizablePanel>
                    </ResizablePanelGroup>
                </ResizablePanel>
            </ResizablePanelGroup>
        </div>
    );
}

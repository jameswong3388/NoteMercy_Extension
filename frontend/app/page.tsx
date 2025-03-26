"use client";

import {FilePond, registerPlugin} from "react-filepond";
import "filepond/dist/filepond.min.css";
import FilePondPluginImageExifOrientation from "filepond-plugin-image-exif-orientation";
import FilePondPluginImagePreview from "filepond-plugin-image-preview";
import "filepond-plugin-image-preview/dist/filepond-plugin-image-preview.css";
import {ResizableHandle, ResizablePanel, ResizablePanelGroup} from "@/components/ui/resizable";
import {ScrollArea} from "@/components/ui/scroll-area";
import {useState} from "react";

// Register FilePond plugins
registerPlugin(FilePondPluginImageExifOrientation, FilePondPluginImagePreview);

export default function Home() {
    const [originalImage, setOriginalImage] = useState<string | null>(null);
    const [processedImage, setProcessedImage] = useState<string | null>(null);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

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
                    <ScrollArea className="h-full p-4">
                        <h2 className="text-lg font-bold mb-4">Handwriting Analysis</h2>
                        <div className="flex items-center justify-center h-[calc(100%-8rem)] text-muted-foreground">
                            Upload an image to see handwriting analysis results
                        </div>
                    </ScrollArea>
                </ResizablePanel>
            </ResizablePanelGroup>
        </div>
    );
}

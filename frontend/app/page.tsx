"use client";

import {FilePond, registerPlugin} from "react-filepond";
import "filepond/dist/filepond.min.css";
import FilePondPluginImageExifOrientation from "filepond-plugin-image-exif-orientation";
import FilePondPluginImagePreview from "filepond-plugin-image-preview";
import "filepond-plugin-image-preview/dist/filepond-plugin-image-preview.css";
import {ResizableHandle, ResizablePanel, ResizablePanelGroup} from "@/components/ui/resizable";
import {ScrollArea} from "@/components/ui/scroll-area";

// Register FilePond plugins
registerPlugin(FilePondPluginImageExifOrientation, FilePondPluginImagePreview);

export default function Home() {
    return (
        <div className="h-screen p-4">
            <ResizablePanelGroup direction="horizontal" className="h-full rounded-lg border">
                <ResizablePanel defaultSize={50} minSize={30}>
                    <ScrollArea className="h-full p-4">
                        <h2 className="text-lg font-bold mb-4">Upload Handwriting</h2>
                        <FilePond
                            allowMultiple={false}
                            acceptedFileTypes={["image/*"]}
                            labelIdle='Drag & Drop your handwriting image or <span class="filepond--label-action">Browse</span>'
                            stylePanelLayout="compact"
                            imagePreviewHeight={256}
                            credits={false}
                        />
                    </ScrollArea>
                </ResizablePanel>
                <ResizableHandle withHandle/>
                <ResizablePanel defaultSize={50} minSize={30}>
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

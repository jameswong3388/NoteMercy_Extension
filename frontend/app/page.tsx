"use client";

import {FilePond, registerPlugin} from "react-filepond";
import "filepond/dist/filepond.min.css";
import FilePondPluginImageExifOrientation from "filepond-plugin-image-exif-orientation";
import FilePondPluginImagePreview from "filepond-plugin-image-preview";
import "filepond-plugin-image-preview/dist/filepond-plugin-image-preview.css";
import {ResizableHandle, ResizablePanel, ResizablePanelGroup} from "@/components/ui/resizable";
import {ScrollArea} from "@/components/ui/scroll-area";
import {useState} from "react";
import {Card, CardContent, CardDescription, CardHeader, CardTitle} from "@/components/ui/card";
import {Tabs, TabsContent, TabsList, TabsTrigger} from "@/components/ui/tabs";
import {Button} from "@/components/ui/button";
import {Dialog, DialogContent, DialogHeader, DialogTitle} from "@/components/ui/dialog";
import {toast} from "sonner";

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
    const [selectedFeature, setSelectedFeature] = useState<string | null>(null);
    const [showFeatureDialog, setShowFeatureDialog] = useState(false);

    const handleFileUpload = async (file: Blob) => {
        try {
            setIsLoading(true);
            setError(null);

            // Show loading toast
            const toastPromise = new Promise<HandwritingFeatures>((resolve, reject) => {
                // Convert file to base64
                const reader = new FileReader();
                reader.onload = async (e) => {
                    const base64Image = e.target?.result as string;
                    setOriginalImage(base64Image);

                    // Remove data URL prefix if present
                    const base64Data = base64Image.split(',')[1] || base64Image;

                    try {
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
                        resolve(data);
                    } catch (error) {
                        reject(error);
                        throw error;
                    }
                };
                reader.readAsDataURL(file);
            });

            toast.promise(toastPromise, {
                loading: 'Processing handwriting image...',
                success: () => 'Handwriting analysis completed successfully!',
                error: (err) => `Error: ${err instanceof Error ? err.message : 'Failed to process image'}`,
            });

        } catch (error) {
            console.error('Error processing image:', error);
            setError(error instanceof Error ? error.message : 'An error occurred while processing the image');
            setProcessedImage(null);
        } finally {
            setIsLoading(false);
        }
    };

    // Function to handle feature selection and display graph
    const handleFeatureSelect = (featureName: string, featureData: any) => {
        setSelectedFeature(featureName);
        setShowFeatureDialog(true);
    };

    // Helper function to format feature name for display
    const formatFeatureName = (name: string) => {
        return name.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
    };

    // Group all features for easy access
    const featureGroups = features ? {
        "Block Lettering": [
            { name: "angularity", data: features.angularity },
            { name: "uppercase_ratio", data: features.uppercase_ratio },
            { name: "pen_pressure", data: features.pen_pressure }
        ],
        "Italic": [
            { name: "vertical_stroke_proportion", data: features.vertical_stroke_proportion },
            { name: "slant_angle", data: features.slant_angle },
            { name: "inter_letter_spacing", data: features.inter_letter_spacing }
        ],
        "Cursive": [
            { name: "stroke_connectivity", data: features.stroke_connectivity },
            { name: "enclosed_loop_ratio", data: features.enclosed_loop_ratio },
            { name: "curvature_continuity", data: features.curvature_continuity }
        ],
        "Calligraphic": [
            { name: "stroke_width_variation", data: features.stroke_width_variation },
            { name: "flourish_extension", data: features.flourish_extension },
            { name: "artistic_consistency", data: features.artistic_consistency }
        ],
        "Shorthand": [
            { name: "stroke_continuity", data: features.stroke_continuity },
            { name: "smooth_curves", data: features.smooth_curves }
        ],
        "Print": [
            { name: "vertical_alignment", data: features.vertical_alignment },
            { name: "letter_size_uniformity", data: features.letter_size_uniformity },
            { name: "discrete_letter", data: features.discrete_letter }
        ]
    } : null;

    // Get selected feature data
    const selectedFeatureData = selectedFeature && features ?
        Object.entries(features).find(([key]) => key === selectedFeature)?.[1] : null;

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
                                {features ? (
                                    <Tabs defaultValue="Block Lettering" className="w-full">
                                        <TabsList className="grid grid-cols-6 mb-4">
                                            {featureGroups && Object.keys(featureGroups).map((group) => (
                                                <TabsTrigger key={group} value={group}>{group}</TabsTrigger>
                                            ))}
                                        </TabsList>

                                        {featureGroups && Object.entries(featureGroups).map(([group, featureList]) => (
                                            <TabsContent key={group} value={group} className="space-y-4">
                                                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                                                    {featureList.map((feature) => (
                                                        <Card
                                                            key={feature.name}
                                                            className="cursor-pointer hover:bg-muted/50 transition-colors"
                                                            onClick={() => handleFeatureSelect(feature.name, feature.data)}
                                                        >
                                                            <CardHeader className="py-3">
                                                                <CardTitle className="text-base">{formatFeatureName(feature.name)}</CardTitle>
                                                            </CardHeader>
                                                            <CardContent className="py-2">
                                                                <pre className="text-xs overflow-hidden text-ellipsis max-h-20">
                                                                    {JSON.stringify(feature.data, null, 2)}
                                                                </pre>
                                                                <Button
                                                                    variant="outline"
                                                                    size="sm"
                                                                    className="mt-2 w-full"
                                                                >
                                                                    View Graph
                                                                </Button>
                                                            </CardContent>
                                                        </Card>
                                                    ))}
                                                </div>
                                            </TabsContent>
                                        ))}
                                    </Tabs>
                                ) : (
                                    <div className="flex items-center justify-center h-[calc(100%-8rem)] text-muted-foreground">
                                        Upload an image to see handwriting analysis results
                                    </div>
                                )}
                            </ScrollArea>
                        </ResizablePanel>
                        <ResizableHandle withHandle />
                        <ResizablePanel defaultSize={50} minSize={30}>
                            <ScrollArea className="h-full p-4">
                                <h2 className="text-lg font-bold mb-4">Handwriting Recognition</h2>
                                {features ? (
                                    <div className="space-y-6">
                                        {/* Handwriting Style Scores */}
                                        <Card>
                                            <CardHeader className="pb-2">
                                                <CardTitle className="text-lg">Handwriting Style Analysis</CardTitle>
                                                <CardDescription>Analysis of your handwriting style characteristics</CardDescription>
                                            </CardHeader>
                                            <CardContent>
                                                <div className="space-y-4">
                                                    {Object.entries(features.handwriting).map(([style, data]) => {
                                                        const score = (data.score * 100);
                                                        // Generate appropriate color based on score
                                                        const getProgressColor = (score: number) => {
                                                            if (score > 75) return "bg-green-500";
                                                            if (score > 50) return "bg-blue-500";
                                                            if (score > 25) return "bg-amber-500";
                                                            return "bg-red-500";
                                                        };
                                                        
                                                        return (
                                                            <div key={style} className="space-y-1">
                                                                <div className="flex justify-between items-center">
                                                                    <span className="capitalize font-medium text-sm">{style.replace(/_/g, ' ')}</span>
                                                                    <span className="font-semibold text-sm">{score.toFixed(1)}%</span>
                                                                </div>
                                                                <div className="w-full h-2 bg-muted rounded-full overflow-hidden">
                                                                    <div 
                                                                        className={`h-full ${getProgressColor(score)} transition-all duration-500 ease-out`}
                                                                        style={{ width: `${score}%` }}
                                                                    ></div>
                                                                </div>
                                                            </div>
                                                        );
                                                    })}
                                                </div>
                                            </CardContent>
                                        </Card>

                                        <Card>
                                            <CardHeader className="pb-2">
                                                <CardTitle className="text-lg">Primary Style</CardTitle>
                                                <CardDescription>Your dominant handwriting
                                                    characteristic</CardDescription>
                                            </CardHeader>
                                            <CardContent>
                                                {(() => {
                                                    // Determine the highest scoring handwriting style
                                                    const sortedStyles = Object.entries(features.handwriting)
                                                        .sort(([, a], [, b]) => b.score - a.score);
                                                    const [topStyle, topData] = sortedStyles[0];
                                                    const numericScore = topData.score * 100;
                                                    const formattedScore = numericScore.toFixed(1);
                                                    const circumference = 2 * Math.PI * 48; // For an SVG circle with radius=48
                                                    const dashOffset = circumference - (numericScore / 100 * circumference);

                                                    // Descriptions for each handwriting style
                                                    const styleDescriptions = {
                                                        block_lettering:
                                                            "Characterized by clear, separated letters with precise angular forms and structured appearance.",
                                                        cursive:
                                                            "Flowing handwriting with connected letters and rhythmic patterns that prioritize speed and efficiency.",
                                                        calligraphic:
                                                            "Artistic writing with varying line thickness, decorative elements, and intentional stylistic flourishes.",
                                                        italic:
                                                            "Slanted writing with a rightward lean and often elongated strokes that create a dynamic appearance.",
                                                        shorthand:
                                                            "Abbreviated writing system designed for rapid note-taking with simplified forms and specialized symbols.",
                                                        print:
                                                            "Neat, standardized letterforms resembling typeset text with consistent spacing and readability.",
                                                    };

                                                    return (
                                                        <div
                                                            className="flex flex-col md:flex-row items-center justify-center gap-8 p-4">
                                                            {/* Circular Progress Indicator */}
                                                            <div className="relative w-32 h-32">
                                                                <svg className="w-full h-full transform -rotate-90">
                                                                    <circle
                                                                        cx="50%"
                                                                        cy="50%"
                                                                        r="48"
                                                                        strokeWidth="8"
                                                                        className="text-muted/30"
                                                                        fill="none"
                                                                    />
                                                                    <circle
                                                                        cx="50%"
                                                                        cy="50%"
                                                                        r="48"
                                                                        strokeWidth="8"
                                                                        className="text-primary"
                                                                        fill="none"
                                                                        style={{
                                                                            strokeDasharray: circumference,
                                                                            strokeDashoffset: dashOffset
                                                                        }}
                                                                    />
                                                                </svg>
                                                                <div
                                                                    className="absolute inset-0 flex items-center justify-center text-xl font-bold">
                                                                    {formattedScore}%
                                                                </div>
                                                            </div>
                                                            {/* Style Name and Description */}
                                                            <div className="text-center md:text-left">
                                                                <div className="text-3xl font-bold capitalize">
                                                                    {topStyle.replace(/_/g, ' ')}
                                                                </div>
                                                                <p className="mt-2 text-muted-foreground">
                                                                    {styleDescriptions[topStyle] ||
                                                                        "A distinctive handwriting style with unique characteristics."}
                                                                </p>
                                                            </div>
                                                        </div>
                                                    );
                                                })()}
                                            </CardContent>
                                        </Card>

                                        <Card>
                                            <CardHeader className="pb-2">
                                                <CardTitle className="text-lg">Visual Characteristics</CardTitle>
                                                <CardDescription>Key features detected in your handwriting</CardDescription>
                                            </CardHeader>
                                            <CardContent>
                                                <div className="grid grid-cols-2 md:grid-cols-3 gap-4 py-2">
                                                    {[
                                                        { name: "Slant", icon: "↗️", value: features.slant_angle?.metrics?.average_angle || "Neutral", description: "Direction of letter tilt" },
                                                        { name: "Pressure", icon: "👆", value: features.pen_pressure?.metrics?.pressure_level || "Medium", description: "Force applied while writing" },
                                                        { name: "Spacing", icon: "⟷", value: features.inter_letter_spacing?.metrics?.spacing_type || "Regular", description: "Space between letters" },
                                                        { name: "Size", icon: "🔍", value: features.letter_size_uniformity?.metrics?.size_category || "Medium", description: "Overall letter size" },
                                                        { name: "Connectivity", icon: "🔗", value: features.stroke_connectivity?.metrics?.connectivity_type || "Mixed", description: "How letters connect" },
                                                        { name: "Consistency", icon: "📏", value: features.artistic_consistency?.metrics?.consistency_level || "Regular", description: "Writing regularity" }
                                                    ].map((item) => (
                                                        <div key={item.name} className="bg-muted/40 rounded-lg p-3 flex flex-col items-center text-center">
                                                            <div className="text-2xl mb-1">{item.icon}</div>
                                                            <h4 className="font-medium text-sm">{item.name}</h4>
                                                            <div className="font-bold text-sm mt-1">{item.value}</div>
                                                            <div className="text-xs text-muted-foreground mt-1">{item.description}</div>
                                                        </div>
                                                    ))}
                                                </div>
                                            </CardContent>
                                        </Card>
                                    </div>
                                ) : (
                                    <div className="flex flex-col items-center justify-center h-[calc(100%-8rem)] text-muted-foreground space-y-4">
                                        <p>Upload an image to see handwriting analysis results</p>
                                    </div>
                                )}
                            </ScrollArea>
                        </ResizablePanel>
                    </ResizablePanelGroup>
                </ResizablePanel>
            </ResizablePanelGroup>

            {/* Feature Detail Dialog */}
            <Dialog open={showFeatureDialog} onOpenChange={setShowFeatureDialog}>
                <DialogContent className="max-w-3xl">
                    <DialogHeader>
                        <DialogTitle>{selectedFeature ? formatFeatureName(selectedFeature) : "Feature Details"}</DialogTitle>
                    </DialogHeader>
                    <div className="mt-4">
                        {selectedFeatureData && (
                            <div className="space-y-4">
                                <div className="bg-muted p-4 rounded-lg">
                                    <h3 className="font-semibold mb-2">Data</h3>
                                    <pre className="text-sm overflow-auto max-h-40">
                                        {JSON.stringify(selectedFeatureData.metrics || selectedFeatureData, null, 2)}
                                    </pre>
                                </div>

                                <div className="bg-card border rounded-lg p-4">
                                    <h3 className="font-semibold mb-4">Graph Visualization</h3>
                                    {selectedFeatureData.graphs && selectedFeatureData.graphs.length > 0 ? (
                                        <div className="flex justify-center">
                                            <img
                                                src={`data:image/png;base64,${selectedFeatureData.graphs[0]}`}
                                                alt={`${selectedFeature} graph`}
                                                className="max-w-full max-h-[400px]"
                                            />
                                        </div>
                                    ) : (
                                        <div className="text-center text-muted-foreground p-8">
                                            No graph data available for this feature.
                                        </div>
                                    )}
                                </div>
                            </div>
                        )}
                    </div>
                </DialogContent>
            </Dialog>
        </div>
    );
}

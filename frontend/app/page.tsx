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
import { PhotoProvider, PhotoView } from "react-photo-view";
import "react-photo-view/dist/react-photo-view.css";

interface HandwritingStyle {
    score: number;
    component_scores?: Record<string, number>;
}

interface HandwritingFeatures {
    handwriting_style_scores: {
        block_lettering: HandwritingStyle;
        cursive: HandwritingStyle;
        calligraphic: HandwritingStyle;
        italic: HandwritingStyle;
        shorthand: HandwritingStyle;
        print: HandwritingStyle;
    };
    analysis_details: {
        block_lettering: {
            angularity: any;
            aspect_ratio: any;
            loop_detection: any;
        };
        italic: {
            vertical_stroke_proportion: any;
            slant_angle: any;
            inter_letter_spacing: any;
        };
        cursive: {
            stroke_connectivity: any;
            enclosed_loop_ratio: any;
            curvature_continuity: any;
            stroke_consistency: any;
        };
        calligraphic: {
            stroke_width_variation: any;
            continuous_part_coverage: any;
            right_angle_corner_detection: any;
        };
        shorthand: {
            stroke_continuity: any;
            smooth_curves: any;
            symbol_density: any;
        };
        print: {
            vertical_alignment: any;
            letter_size_uniformity: any;
            discrete_letter: any;
        };
    };
}

// Register FilePond plugins
registerPlugin(FilePondPluginImageExifOrientation, FilePondPluginImagePreview);

export default function Home() {
    const [originalImage, setOriginalImage] = useState<string | null>(null);
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
        } finally {
            setIsLoading(false);
        }
    };

    // Function to handle feature selection and display graph
    const handleFeatureSelect = (featureName: string, featureData: any, group: string) => {
        setSelectedFeature(`${group}.${featureName}`);
        setShowFeatureDialog(true);
    };

    // Helper function to format feature name for display
    const formatFeatureName = (name: string) => {
        return name.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
    };

    // Group all features for easy access
    const featureGroups = features ? {
        "Block Lettering": [
            { name: "angularity", data: features.analysis_details.block_lettering.angularity },
            { name: "aspect_ratio", data: features.analysis_details.block_lettering.aspect_ratio },
            { name: "loop_detection", data: features.analysis_details.block_lettering.loop_detection }
        ],
        "Italic": [
            { name: "vertical_stroke_proportion", data: features.analysis_details.italic.vertical_stroke_proportion },
            { name: "slant_angle", data: features.analysis_details.italic.slant_angle },
            { name: "inter_letter_spacing", data: features.analysis_details.italic.inter_letter_spacing }
        ],
        "Cursive": [
            { name: "stroke_connectivity", data: features.analysis_details.cursive.stroke_connectivity },
            { name: "enclosed_loop_ratio", data: features.analysis_details.cursive.enclosed_loop_ratio },
            { name: "curvature_continuity", data: features.analysis_details.cursive.curvature_continuity },
            { name: "stroke_consistency", data: features.analysis_details.cursive.stroke_consistency }
        ],
        "Calligraphic": [
            { name: "stroke_width_variation", data: features.analysis_details.calligraphic.stroke_width_variation },
            { name: "continuous_part_coverage", data: features.analysis_details.calligraphic.continuous_part_coverage },
            { name: "right_angle_corner_detection", data: features.analysis_details.calligraphic.right_angle_corner_detection }
        ],
        "Shorthand": [
            { name: "stroke_continuity", data: features.analysis_details.shorthand.stroke_continuity },
            { name: "smooth_curves", data: features.analysis_details.shorthand.smooth_curves },
            { name: "symbol_density", data: features.analysis_details.shorthand.symbol_density }
        ],
        "Print": [
            { name: "vertical_alignment", data: features.analysis_details.print.vertical_alignment },
            { name: "letter_size_uniformity", data: features.analysis_details.print.letter_size_uniformity },
            { name: "discrete_letter", data: features.analysis_details.print.discrete_letter }
        ]
    } : null;

    // Get selected feature data
    const getSelectedFeatureData = () => {
        if (!selectedFeature || !features) return null;
        
        const [group, feature] = selectedFeature.split('.');
        if (!group || !feature) return null;
        
        const groupKey = group.toLowerCase().replace(/ /g, '_') as keyof typeof features.analysis_details;
        const featureGroup = features.analysis_details[groupKey];
        
        if (!featureGroup) return null;
        
        // Type assertion to handle feature access
        return (featureGroup as Record<string, any>)[feature] || null;
    };
    
    const selectedFeatureData = getSelectedFeatureData();

    return (
        <div className="h-screen p-4">
            <PhotoProvider>
                <ResizablePanelGroup direction="horizontal" className="h-full rounded-lg border">
                    <ResizablePanel defaultSize={30} minSize={30}>
                        <ResizablePanelGroup direction="vertical" className="h-full">
                            <ResizablePanel defaultSize={30} minSize={30}>
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
                            <ResizablePanel defaultSize={70} minSize={40}>
                                <ScrollArea className="h-full p-4">
                                    <h2 className="text-lg font-bold mb-4">Original Image</h2>
                                    <div className="flex items-center justify-center h-[calc(100%-2rem)]">
                                        {originalImage ? (
                                            <PhotoView src={originalImage}>
                                                <img src={originalImage} alt="Original" className="max-w-full max-h-full cursor-zoom-in" />
                                            </PhotoView>
                                        ) : (
                                            <span className="text-muted-foreground">Original image will appear here</span>
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
                                    {isLoading ? (
                                        <div className="flex items-center justify-center h-[calc(100%-2rem)]">
                                            <span className="text-muted-foreground">Processing image...</span>
                                        </div>
                                    ) : error ? (
                                        <div className="flex items-center justify-center h-[calc(100%-2rem)]">
                                            <span className="text-red-500">{error}</span>
                                        </div>
                                    ) : features ? (
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
                                                                onClick={() => handleFeatureSelect(feature.name, feature.data, group)}
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
                                                    <div className="space-y-6">
                                                        {Object.entries(features.handwriting_style_scores).map(([style, data]) => {
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
                                                                    <div 
                                                                        className="relative group cursor-pointer"
                                                                        title="Hover for component scores"
                                                                    >
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
                                                                        
                                                                        {data.component_scores && Object.keys(data.component_scores).length > 0 && (
                                                                            <div className="absolute right-0 top-full mt-1 w-64 z-10 bg-card shadow-lg rounded-md p-3 border opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all duration-200 pointer-events-none">
                                                                                <h4 className="text-xs font-medium mb-2">Component Scores:</h4>
                                                                                <div className="grid grid-cols-1 gap-y-1.5 text-xs">
                                                                                    {Object.entries(data.component_scores).map(([componentName, componentScore]) => {
                                                                                        const formattedScore = (componentScore as number * 100).toFixed(1);
                                                                                        return (
                                                                                            <div key={componentName} className="flex justify-between">
                                                                                                <span className="text-muted-foreground capitalize">
                                                                                                    {componentName.replace(/_/g, ' ')}:
                                                                                                </span>
                                                                                                <span className="font-medium">{formattedScore}%</span>
                                                                                            </div>
                                                                                        );
                                                                                    })}
                                                                                </div>
                                                                            </div>
                                                                        )}
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
                                                        const sortedStyles = Object.entries(features.handwriting_style_scores)
                                                            .sort(([, a], [, b]) => b.score - a.score);
                                                        const [topStyle, topData] = sortedStyles[0];
                                                        const numericScore = topData.score * 100;
                                                        const formattedScore = numericScore.toFixed(1);
                                                        const circumference = 2 * Math.PI * 48; // For an SVG circle with radius=48
                                                        const dashOffset = circumference - (numericScore / 100 * circumference);

                                                        // Descriptions for each handwriting style
                                                        const styleDescriptions: Record<string, string> = {
                                                            block_lettering:
                                                                "All uppercase, bold, rigid structure.",
                                                            cursive:
                                                                "Flowing, connected letters with loops.",
                                                            calligraphic:
                                                                "Artistic strokes, exaggerated flourishes.",
                                                            italic:
                                                                "Slanted and semi-connected strokes.",
                                                            shorthand:
                                                                "Abbreviated strokes, symbols, and ligatures",
                                                            print:
                                                                "Clearly separated letters, often uniform",
                                                        };

                                                        return (
                                                            <div
                                                                className="flex flex-col md:flex-row items-center justify-center gap-8 p-4">
                                                                {/* Circular Progress Indicator */}
                                                                <div className="relative w-32 h-32 border-r-2">
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
                    <DialogContent className="max-w-4xl max-h-[90vh]">
                        <PhotoProvider>
                            <DialogHeader>
                                <DialogTitle>
                                    {selectedFeature ? formatFeatureName(selectedFeature.split('.')[1] || '') : "Feature Details"}
                                </DialogTitle>
                            </DialogHeader>
                            <ScrollArea className="mt-4 max-h-[calc(90vh-80px)]">
                                {selectedFeatureData && (
                                    <div className="space-y-6">
                                        {/* 1. Preprocessed Image Section (if available) */}
                                        {selectedFeatureData.preprocessed_image && (
                                            <div className="bg-card border rounded-lg p-4">
                                                <h3 className="font-semibold mb-4">Preprocessed Image</h3>
                                                <div className="flex justify-center overflow-hidden">
                                                    <PhotoView src={`data:image/png;base64,${selectedFeatureData.preprocessed_image}`}>
                                                        <img
                                                            src={`data:image/png;base64,${selectedFeatureData.preprocessed_image}`}
                                                            alt="Preprocessed"
                                                            className="max-w-full object-contain max-h-[300px] cursor-zoom-in"
                                                        />
                                                    </PhotoView>
                                                </div>
                                            </div>
                                        )}
                                        
                                        {/* 2. Data Section */}
                                        <div className="bg-muted p-4 rounded-lg">
                                            <h3 className="font-semibold mb-2">Data</h3>
                                            <pre className="text-sm overflow-auto max-h-60 whitespace-pre-wrap break-words">
                                                {JSON.stringify(selectedFeatureData.metrics || selectedFeatureData, null, 2)}
                                            </pre>
                                        </div>

                                        {/* 3. Graph Visualization Section */}
                                        <div className="bg-card border rounded-lg p-4">
                                            <h3 className="font-semibold mb-4">Graph Visualization</h3>
                                            {selectedFeatureData.graphs && selectedFeatureData.graphs.length > 0 ? (
                                                <div className="flex justify-center overflow-hidden">
                                                    <PhotoView src={`data:image/png;base64,${selectedFeatureData.graphs[0]}`}>
                                                        <img
                                                            src={`data:image/png;base64,${selectedFeatureData.graphs[0]}`}
                                                            alt={`${selectedFeature?.split('.')[1]} graph`}
                                                            className="max-w-full object-contain max-h-[300px] cursor-zoom-in"
                                                        />
                                                    </PhotoView>
                                                </div>
                                            ) : (
                                                <div className="text-center text-muted-foreground p-8">
                                                    No graph data available for this feature.
                                                </div>
                                            )}
                                        </div>
                                    </div>
                                )}
                            </ScrollArea>
                        </PhotoProvider>
                    </DialogContent>
                </Dialog>
            </PhotoProvider>
        </div>
    );
}

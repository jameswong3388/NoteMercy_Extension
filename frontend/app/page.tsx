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
import {PhotoProvider, PhotoView} from "react-photo-view";
import "react-photo-view/dist/react-photo-view.css";
import {cn} from "@/lib/utils";

interface HandwritingStyle {
    score: number;
    component_scores?: Record<string, number>;
}

interface FeatureData {
    data: {
        metrics: Record<string, any>;
        graphs: string[];
        preprocessed_image: string;
    };
    is_dominant: boolean;
    is_shared: boolean;
    weightage: number;
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
            angularity: FeatureData;
            aspect_ratio: FeatureData;
            loop_detection: FeatureData;
        };
        italic: {
            slant_angle: FeatureData;
            vertical_stroke_proportion: FeatureData;
            inter_letter_spacing: FeatureData;
        };
        cursive: {
            stroke_connectivity: FeatureData;
            curvature_continuity: FeatureData;
            enclosed_loop_ratio: FeatureData;
            stroke_consistency: FeatureData;
        };
        calligraphic: {
            stroke_width_variation: FeatureData;
            right_angle_corner_detection: FeatureData;
            continuous_part_coverage: FeatureData;
        };
        shorthand: {
            stroke_terminal: FeatureData;
            symbol_density: FeatureData;
            curve_smoothness: FeatureData;
        };
        print: {
            discrete_letter: FeatureData;
            vertical_alignment: FeatureData;
            letter_size_uniformity: FeatureData;
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
    const [activeTab, setActiveTab] = useState("Block Lettering");

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
                            body: JSON.stringify({image: base64Data}),
                        });

                        if (!response.ok) {
                            const errorData = await response.json().catch(() => ({}));
                            throw new Error(errorData.detail || 'Failed to process image');
                        }

                        const responseData = await response.json();

                        // Fix for the nested data structure in API response
                        let data: HandwritingFeatures;

                        // Check if data is nested inside an object with a curly brace
                        if (responseData && typeof responseData === 'object' && Object.keys(responseData).length === 1 &&
                            Object.keys(responseData)[0] === '0') {
                            // The API is returning a nested object with a numeric key
                            data = responseData['0'];
                        } else {
                            // Use the response data directly if it's already in the expected format
                            data = responseData;
                        }

                        setFeatures(data);

                        // Determine the highest scoring style and set it as the active tab
                        const styleScores = data.handwriting_style_scores;
                        const topStyle = Object.entries(styleScores)
                            .sort(([, a], [, b]) => b.score - a.score)[0][0];

                        // Map the style key to tab name
                        const styleToTabMap: Record<string, string> = {
                            block_lettering: "Block Lettering",
                            cursive: "Cursive",
                            calligraphic: "Calligraphic",
                            italic: "Italic",
                            shorthand: "Shorthand",
                            print: "Print"
                        };

                        setActiveTab(styleToTabMap[topStyle] || "Block Lettering");
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
            {name: "angularity", data: features.analysis_details.block_lettering.angularity},
            {name: "aspect_ratio", data: features.analysis_details.block_lettering.aspect_ratio},
            {name: "loop_detection", data: features.analysis_details.block_lettering.loop_detection}
        ],
        "Calligraphic": [
            {name: "stroke_width_variation", data: features.analysis_details.calligraphic.stroke_width_variation},
            {name: "right_angle_corner_detection", data: features.analysis_details.calligraphic.right_angle_corner_detection},
            {name: "continuous_part_coverage", data: features.analysis_details.calligraphic.continuous_part_coverage}
        ],
        "Cursive": [
            {name: "stroke_connectivity", data: features.analysis_details.cursive.stroke_connectivity},
            {name: "curvature_continuity", data: features.analysis_details.cursive.curvature_continuity},
            {name: "enclosed_loop_ratio", data: features.analysis_details.cursive.enclosed_loop_ratio},
            {name: "stroke_consistency", data: features.analysis_details.cursive.stroke_consistency}
        ],
        "Italic": [
            {name: "slant_angle", data: features.analysis_details.italic.slant_angle},
            {name: "vertical_stroke_proportion", data: features.analysis_details.italic.vertical_stroke_proportion},
            {name: "inter_letter_spacing", data: features.analysis_details.italic.inter_letter_spacing}
        ],
        "Print": [
            {name: "discrete_letter", data: features.analysis_details.print.discrete_letter},
            {name: "vertical_alignment", data: features.analysis_details.print.vertical_alignment},
            {name: "letter_size_uniformity", data: features.analysis_details.print.letter_size_uniformity},
        ],
        "Shorthand": [
            {name: "stroke_terminal", data: features.analysis_details.shorthand.stroke_terminal},
            {name: "symbol_density", data: features.analysis_details.shorthand.symbol_density},
            {name: "curve_smoothness", data: features.analysis_details.shorthand.curve_smoothness}
        ],
    } : null;

    // Get selected feature data
    const getSelectedFeatureData = () => {
        if (!selectedFeature || !features) return null;

        const [group, feature] = selectedFeature.split('.');
        if (!group || !feature) return null;

        const groupKey = group.toLowerCase().replace(/ /g, '_') as keyof typeof features.analysis_details;
        const featureGroup = features.analysis_details[groupKey];

        if (!featureGroup) return null;

        // Get the feature object
        const featureObj = (featureGroup as Record<string, any>)[feature];
        if (!featureObj) return null;

        return featureObj;
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
                            <ResizableHandle withHandle/>
                            <ResizablePanel defaultSize={70} minSize={40}>
                                <ScrollArea className="h-full p-4">
                                    <h2 className="text-lg font-bold mb-4">Original Image</h2>
                                    <div className="flex items-center justify-center h-[calc(100%-2rem)]">
                                        {originalImage ? (
                                            <PhotoView src={originalImage}>
                                                <img src={originalImage} alt="Original"
                                                     className="max-w-full max-h-full cursor-zoom-in"/>
                                            </PhotoView>
                                        ) : (
                                            <span
                                                className="text-muted-foreground">Original image will appear here</span>
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
                                        <Tabs
                                            value={activeTab}
                                            onValueChange={setActiveTab}
                                            className="w-full"
                                        >
                                            <TabsList className="grid grid-cols-6 mb-4">
                                                {featureGroups && Object.keys(featureGroups).map((group) => (
                                                    <TabsTrigger key={group} value={group}>{group}</TabsTrigger>
                                                ))}
                                            </TabsList>

                                            {featureGroups && Object.entries(featureGroups).map(([group, featureList]) => (
                                                <TabsContent
                                                    key={group}
                                                    value={group}
                                                    className={cn(
                                                        "space-y-4",
                                                        activeTab === group ? "animate-tab-slide-in" : ""
                                                    )}
                                                >
                                                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                                                        {featureList.map((feature) => (
                                                            <Card
                                                                key={feature.name}
                                                                className="cursor-pointer hover:bg-muted/50 transition-colors"
                                                                onClick={() => handleFeatureSelect(feature.name, feature.data, group)}
                                                            >
                                                                <CardHeader className="py-3">
                                                                    <CardTitle
                                                                        className="text-base flex justify-between items-start">
                                                                        <span>{formatFeatureName(feature.name)}</span>
                                                                        <div className="flex gap-1 flex-wrap">
                                                                            {(feature.data.is_dominant) && (
                                                                                <span
                                                                                    className="inline-flex items-center rounded-md bg-blue-50 px-2 py-1 text-xs font-medium text-blue-700 ring-1 ring-inset ring-blue-700/10">
                                                                                    Dominant
                                                                                </span>
                                                                            )}
                                                                            {(feature.data.is_shared) && (
                                                                                <span
                                                                                    className="inline-flex items-center rounded-md bg-green-50 px-2 py-1 text-xs font-medium text-green-700 ring-1 ring-inset ring-green-700/10">
                                                                                    Shared
                                                                                </span>
                                                                            )}
                                                                            <span
                                                                                className="inline-flex items-center rounded-md bg-gray-50 px-2 py-1 text-xs font-medium text-gray-700 ring-1 ring-inset ring-gray-700/10">
                                                                                W: {feature.data.weightage.toFixed(1)}
                                                                            </span>
                                                                        </div>
                                                                    </CardTitle>
                                                                </CardHeader>
                                                                <CardContent className="py-2">
                                                                    <pre
                                                                        className="text-xs overflow-hidden text-ellipsis max-h-20">
                                                                        {JSON.stringify((feature.data as unknown as {
                                                                            data: { metrics: Record<string, any> }
                                                                        })?.data?.metrics || {}, null, 2)}
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
                                        <div
                                            className="flex items-center justify-center h-[calc(100%-8rem)] text-muted-foreground">
                                            Upload an image to see handwriting analysis results
                                        </div>
                                    )}
                                </ScrollArea>
                            </ResizablePanel>
                            <ResizableHandle withHandle/>
                            <ResizablePanel defaultSize={50} minSize={30}>
                                <ScrollArea className="h-full p-4">
                                    <h2 className="text-lg font-bold mb-4">Handwriting Recognition</h2>
                                    {features ? (
                                        <div className="space-y-6">
                                            {/* Handwriting Style Scores */}
                                            <Card className="animate-tab-fade-in">
                                                <CardHeader className="pb-2">
                                                    <CardTitle className="text-lg">Handwriting Style
                                                        Analysis</CardTitle>
                                                    <CardDescription>Analysis of your handwriting style
                                                        characteristics</CardDescription>
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
                                                                        <div
                                                                            className="flex justify-between items-center">
                                                                            <span
                                                                                className="capitalize font-medium text-sm">{style.replace(/_/g, ' ')}</span>
                                                                            <span
                                                                                className="font-semibold text-sm">{score.toFixed(1)}%</span>
                                                                        </div>
                                                                        <div
                                                                            className="w-full h-2 bg-muted rounded-full overflow-hidden">
                                                                            <div
                                                                                className={`h-full ${getProgressColor(score)} transition-all duration-500 ease-out`}
                                                                                style={{width: `${score}%`}}
                                                                            ></div>
                                                                        </div>

                                                                        {data.component_scores && Object.keys(data.component_scores).length > 0 && (
                                                                            <div
                                                                                className="absolute right-0 bottom-full mb-1 w-64 z-50 bg-card shadow-lg rounded-md p-3 border opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all duration-200 pointer-events-none">
                                                                                <h4 className="text-xs font-medium mb-2">Component
                                                                                    Scores:</h4>
                                                                                <div
                                                                                    className="grid grid-cols-1 gap-y-1.5 text-xs">
                                                                                    {Object.entries(data.component_scores).map(([componentName, componentScore]) => {
                                                                                        const formattedScore = (componentScore as number * 100).toFixed(1);
                                                                                        return (
                                                                                            <div key={componentName}
                                                                                                 className="flex justify-between">
                                                                                                <span
                                                                                                    className="text-muted-foreground capitalize">
                                                                                                    {componentName.replace(/_/g, ' ')}:
                                                                                                </span>
                                                                                                <span
                                                                                                    className="font-medium">{formattedScore}%</span>
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

                                            <Card className="animate-tab-fade-in" style={{animationDelay: "0.1s"}}>
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
                                        <div
                                            className="flex flex-col items-center justify-center h-[calc(100%-8rem)] text-muted-foreground space-y-4">
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
                                <DialogTitle className="flex justify-between items-start">
                                    <span>{selectedFeature ? formatFeatureName(selectedFeature.split('.')[1] || '') : "Feature Details"}</span>
                                    {selectedFeatureData && (
                                        <div className="flex gap-1 flex-wrap">
                                            {selectedFeatureData.is_dominant && (
                                                <span
                                                    className="inline-flex items-center rounded-md bg-blue-50 px-2 py-1 text-xs font-medium text-blue-700 ring-1 ring-inset ring-blue-700/10">
                                                    Dominant
                                                </span>
                                            )}
                                            {selectedFeatureData.is_shared && (
                                                <span
                                                    className="inline-flex items-center rounded-md bg-green-50 px-2 py-1 text-xs font-medium text-green-700 ring-1 ring-inset ring-green-700/10">
                                                    Shared
                                                </span>
                                            )}
                                            <span
                                                className="inline-flex items-center rounded-md bg-gray-50 px-2 py-1 text-xs font-medium text-gray-700 ring-1 ring-inset ring-gray-700/10">
                                                W: {selectedFeatureData.weightage.toFixed(1)}
                                            </span>
                                        </div>
                                    )}
                                </DialogTitle>
                            </DialogHeader>
                            <ScrollArea className="mt-4 max-h-[calc(90vh-80px)]">
                                {selectedFeatureData && (
                                    <div className="space-y-6 animate-tab-fade-in">
                                        {/* 1. Preprocessed Image Section (if available) */}
                                        {(selectedFeatureData as unknown as {
                                            data: { preprocessed_image: string }
                                        })?.data?.preprocessed_image && (
                                            <div className="bg-card border rounded-lg p-4">
                                                <h3 className="font-semibold mb-4">Preprocessed Image</h3>
                                                <div className="flex justify-center overflow-hidden">
                                                    <PhotoView
                                                        src={`data:image/png;base64,${(selectedFeatureData as unknown as {
                                                            data: { preprocessed_image: string }
                                                        }).data.preprocessed_image}`}>
                                                        <img
                                                            src={`data:image/png;base64,${(selectedFeatureData as unknown as {
                                                                data: { preprocessed_image: string }
                                                            }).data.preprocessed_image}`}
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
                                            <pre
                                                className="text-sm overflow-auto max-h-60 whitespace-pre-wrap break-words">
                                                {JSON.stringify(
                                                    (selectedFeatureData as unknown as {
                                                        data: { metrics: Record<string, any> }
                                                    })?.data?.metrics || {},
                                                    null, 2
                                                )}
                                            </pre>
                                        </div>

                                        {/* 3. Graph Visualization Section */}
                                        <div className="bg-card border rounded-lg p-4">
                                            <h3 className="font-semibold mb-4">Graph Visualization</h3>
                                            {(selectedFeatureData as unknown as {
                                                data: { graphs: string[] }
                                            })?.data?.graphs &&
                                            (selectedFeatureData as unknown as {
                                                data: { graphs: string[] }
                                            }).data.graphs.length > 0 ? (
                                                <div className="grid grid-cols-1 gap-6">
                                                    {(selectedFeatureData as unknown as { data: { graphs: string[] } }).data.graphs.map((graph, index) => (
                                                        <div key={index} className="flex justify-center overflow-hidden">
                                                            <PhotoView src={`data:image/png;base64,${graph}`}>
                                                                <img
                                                                    src={`data:image/png;base64,${graph}`}
                                                                    alt={`${selectedFeature?.split('.')[1]} graph ${index + 1}`}
                                                                    className="max-w-full object-contain max-h-[300px] cursor-zoom-in"
                                                                />
                                                            </PhotoView>
                                                        </div>
                                                    ))}
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

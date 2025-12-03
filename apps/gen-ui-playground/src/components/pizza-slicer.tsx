"use client";

import { useState } from "react";
import { Slider } from "@/components/ui/slider";
import { Button } from "@/components/ui/button";

interface PizzaSlicerProps {
  initialSlices?: number;
}

export function PizzaSlicer({ initialSlices = 1 }: PizzaSlicerProps) {
  const [slices, setSlices] = useState(initialSlices);

  // Calculate paths for slices
  const radius = 100;
  const center = 100;
  
  const getCoordinatesForPercent = (percent: number) => {
    const x = center + radius * Math.cos(2 * Math.PI * percent);
    const y = center + radius * Math.sin(2 * Math.PI * percent);
    return [x, y];
  };

  const slicePaths = [];
  if (slices > 1) {
    for (let i = 0; i < slices; i++) {
        const startPercent = i / slices;
        const endPercent = (i + 1) / slices;
        const [startX, startY] = getCoordinatesForPercent(startPercent);
        const [endX, endY] = getCoordinatesForPercent(endPercent);
        const largeArcFlag = endPercent - startPercent > 0.5 ? 1 : 0;
        const pathData = [
            `M ${center} ${center}`,
            `L ${startX} ${startY}`,
            `A ${radius} ${radius} 0 ${largeArcFlag} 1 ${endX} ${endY}`,
            `Z`
        ].join(' ');
        slicePaths.push(pathData);
    }
  } else {
      // Whole pizza
      slicePaths.push(`M ${center} ${center - radius} A ${radius} ${radius} 0 1 1 ${center} ${center + radius} A ${radius} ${radius} 0 1 1 ${center} ${center - radius} Z`);
  }

  return (
    <div className="p-6 border rounded-xl shadow-sm bg-card w-full max-w-md mx-auto space-y-6">
      <div className="text-center space-y-2">
        <h3 className="text-lg font-semibold">🍕 Pizza Slicer</h3>
        <p className="text-sm text-muted-foreground">
          Cutting into <span className="font-bold text-primary">{slices}</span> slices
        </p>
      </div>

      <div className="flex justify-center">
        <svg width="200" height="200" viewBox="0 0 200 200" className="transform -rotate-90">
          <circle cx="100" cy="100" r="100" fill="#e2e8f0" />
          {slicePaths.map((path, i) => (
            <path
              key={i}
              d={path}
              fill="#fcd34d"
              stroke="#fff"
              strokeWidth="2"
              className="transition-all duration-300 hover:fill-orange-400"
            />
          ))}
          {/* Crust */}
          <circle cx="100" cy="100" r="95" fill="none" stroke="#d97706" strokeWidth="4" opacity="0.5" />
          {/* Pepperoni (random-ish positions) */}
          <circle cx="60" cy="80" r="8" fill="#ef4444" opacity="0.8" />
          <circle cx="140" cy="120" r="8" fill="#ef4444" opacity="0.8" />
          <circle cx="120" cy="60" r="8" fill="#ef4444" opacity="0.8" />
          <circle cx="80" cy="150" r="8" fill="#ef4444" opacity="0.8" />
        </svg>
      </div>

      <div className="space-y-4">
        <Slider
          value={[slices]}
          onValueChange={(v) => setSlices(v[0])}
          min={1}
          max={12}
          step={1}
          className="w-full"
        />
        <div className="flex justify-between text-xs text-muted-foreground">
            <span>1 slice</span>
            <span>12 slices</span>
        </div>
      </div>
      
      <div className="flex justify-center gap-2">
          <Button variant="outline" size="sm" onClick={() => setSlices(Math.max(1, slices - 1))}>-</Button>
          <Button variant="outline" size="sm" onClick={() => setSlices(Math.min(12, slices + 1))}>+</Button>
      </div>
    </div>
  );
}

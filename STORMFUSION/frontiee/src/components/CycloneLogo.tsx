import React from 'react';
import stormFusionLogo from '../assets/stormfusion-cyclone-logo.jpg';

interface CycloneLogoProps {
  size?: number | string;
  className?: string;
  animated?: boolean;
}

export const CycloneLogo: React.FC<CycloneLogoProps> = ({
  size = 36,
  className = '',
  animated = false
}) => {
  const dimensionStyle = typeof size === 'number' 
    ? { width: `${size}px`, height: `${size}px` } 
    : { width: size, height: size };

  return (
    <div
      style={dimensionStyle}
      className={`relative inline-flex items-center justify-center shrink-0 rounded-full overflow-hidden bg-slate-950 ring-1 ring-cyan-500/40 shadow-xs select-none ${className}`}
      aria-label="StormFusion Cyclone Vortex Logo"
    >
      <img
        src={stormFusionLogo}
        alt="StormFusion Cyclone Vortex"
        className={`w-full h-full object-cover rounded-full ${
          animated ? 'cyclone-logo-spin' : ''
        }`}
        loading="eager"
        decoding="async"
      />
    </div>
  );
};

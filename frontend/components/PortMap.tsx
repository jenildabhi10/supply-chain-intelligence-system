"use client";

import { MapContainer, TileLayer, CircleMarker, Tooltip } from "react-leaflet";
import { useRouter } from "next/navigation";
import type { RiskSummary, PortInfo } from "@/lib/api";
import { tierColor } from "@/lib/utils";
import { useMemo } from "react";

interface Props {
  ports: PortInfo[];
  summaries: RiskSummary[];
  height?: number;
}

export default function PortMap({ ports, summaries, height = 460 }: Props) {
  const router = useRouter();
  const summaryById = useMemo(
    () => Object.fromEntries(summaries.map((s) => [s.port_id, s])),
    [summaries],
  );

  const center: [number, number] = [37.0, -96.0];

  return (
    <div
      className="w-full overflow-hidden rounded-xl border border-border"
      style={{ height }}
    >
      <MapContainer
        center={center}
        zoom={4}
        scrollWheelZoom={false}
        style={{ height: "100%", width: "100%" }}
      >
        <TileLayer
          url="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png"
          attribution='&copy; OpenStreetMap, &copy; CARTO'
        />
        {ports.map((p) => {
          const s = summaryById[p.port_id];
          const score = s?.risk_score ?? 0;
          const color = tierColor(s?.risk_tier ?? "low");
          const radius = 8 + (score / 100) * 18;
          return (
            <CircleMarker
              key={p.port_id}
              center={[p.lat, p.lon]}
              radius={radius}
              pathOptions={{
                color,
                weight: 2,
                fillColor: color,
                fillOpacity: 0.35,
              }}
              eventHandlers={{
                click: () => router.push(`/ports/${p.port_id}`),
              }}
            >
              <Tooltip direction="top" offset={[0, -radius]} opacity={1}>
                <div className="font-sans">
                  <div className="font-medium text-[13px]">{p.name}</div>
                  <div className="text-[11px] text-zinc-600">
                    {s ? `Risk ${score.toFixed(1)} · ${s.risk_tier}` : "no data"}
                  </div>
                </div>
              </Tooltip>
            </CircleMarker>
          );
        })}
      </MapContainer>
    </div>
  );
}

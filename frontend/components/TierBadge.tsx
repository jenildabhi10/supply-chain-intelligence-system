import { cn, tierClass } from "@/lib/utils";

interface Props {
  tier: string;
  size?: "sm" | "md";
  className?: string;
}

export function TierBadge({ tier, size = "sm", className }: Props) {
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1.5 rounded-md border font-mono uppercase tracking-wider",
        tierClass(tier),
        size === "sm" ? "px-2 py-0.5 text-[10px]" : "px-3 py-1 text-xs",
        className,
      )}
    >
      <span
        className="w-1.5 h-1.5 rounded-full"
        style={{ background: "currentColor" }}
      />
      {tier}
    </span>
  );
}

// components/chat/MetricBadge.tsx

interface Props {
  label: string;
  value: string;
  type: "high" | "med" | "low" | "text";
}

export default function MetricBadge({ label, value, type }: Props) {
  return (
    <div className="metric-badge">
      <span className="metric-label">{label}</span>
      <span className={`metric-value metric-value--${type}`}>{value}</span>
    </div>
  );
}

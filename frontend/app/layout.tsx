import type { Metadata } from "next";
import { Inter, JetBrains_Mono } from "next/font/google";
import "./globals.css";
import "leaflet/dist/leaflet.css";
import { TopNav } from "@/components/TopNav";

const inter = Inter({
  variable: "--font-inter",
  subsets: ["latin"],
  display: "swap",
});

const jetbrains = JetBrains_Mono({
  variable: "--font-jetbrains",
  subsets: ["latin"],
  display: "swap",
});

export const metadata: Metadata = {
  title: "Supply Chain Intelligence",
  description:
    "Real-time port disruption risk intelligence for U.S. supply chains. Evidence-grounded, model-explained, agent-narrated.",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" className={`${inter.variable} ${jetbrains.variable}`}>
      <body suppressHydrationWarning className="min-h-screen flex flex-col bg-bg text-fg">
        <TopNav />
        <main className="flex-1 w-full max-w-[1400px] mx-auto px-6 md:px-10 py-10">
          {children}
        </main>
        <footer className="border-t border-border mt-12">
          <div className="max-w-[1400px] mx-auto px-6 md:px-10 py-6 flex flex-col md:flex-row gap-2 md:gap-6 items-start md:items-center justify-between text-xs text-fg-dim font-mono">
            <span>{"// supply_chain_intelligence.system"}</span>
            <span>Built with Next.js · FastAPI · XGBoost · Groq</span>
          </div>
        </footer>
      </body>
    </html>
  );
}

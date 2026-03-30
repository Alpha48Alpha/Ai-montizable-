import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "AI-Monetizable — Animation Movie Engine",
  description:
    "Full animation movie engine: generate complete 10-section production packages for any movie, anime, cartoon, or trailer.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className="bg-gray-950 text-gray-100 antialiased">{children}</body>
    </html>
  );
}

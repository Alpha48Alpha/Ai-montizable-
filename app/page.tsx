export default function HomePage() {
  const sections = [
    { number: 1, name: "Title", description: "Working title and tagline" },
    { number: 2, name: "Runtime", description: "Total duration and per-scene time-codes" },
    { number: 3, name: "Characters", description: "Cast list with appearance, personality, and role" },
    { number: 4, name: "Scene List", description: "Ordered scenes with time-codes and location" },
    { number: 5, name: "Dialogue", description: "Full script with speaker labels" },
    { number: 6, name: "Subtitles", description: "SRT-format blocks with timestamps" },
    { number: 7, name: "Visual Prompts", description: "Per-scene AI generation prompts" },
    { number: 8, name: "Audio Plan", description: "Music, SFX, and narration guidance" },
    { number: 9, name: "Assembly Plan", description: "Editing order, transitions, export settings" },
    { number: 10, name: "Deliverables", description: "Output file list with quality labels" },
  ];

  const qualityLabels = [
    { label: "prototype animation", description: "Rough motion test; limited frames, placeholder art" },
    { label: "concept clip", description: "Style/mood exploration; not final designs" },
    { label: "motion video", description: "Animatic or motion-graphics with basic movement" },
    { label: "slideshow film", description: "Sequential stills with transitions and audio" },
    { label: "rendered short", description: "Full frame-by-frame rendered animation" },
  ];

  return (
    <main className="min-h-screen">
      {/* Hero */}
      <section className="bg-gradient-to-b from-indigo-950 to-gray-950 px-6 py-20 text-center">
        <div className="mx-auto max-w-3xl">
          <p className="mb-3 text-sm font-semibold uppercase tracking-widest text-indigo-400">
            Status: Active
          </p>
          <h1 className="mb-4 text-4xl font-extrabold tracking-tight sm:text-5xl">
            🎬 AI-Monetizable
          </h1>
          <p className="mb-2 text-xl font-semibold text-indigo-300">
            Full Animation Movie Engine
          </p>
          <p className="text-gray-400">
            Every movie, video, anime, cartoon, and trailer request is treated as
            a full production task — automatically generating a complete
            10-section production package.
          </p>
          <div className="mt-8 flex flex-wrap justify-center gap-3">
            <a
              href="https://github.com/Alpha48Alpha/Ai-montizable-"
              className="rounded-lg bg-indigo-600 px-5 py-2.5 text-sm font-semibold text-white hover:bg-indigo-500 transition-colors"
            >
              View on GitHub
            </a>
            <a
              href="#get-started"
              className="rounded-lg border border-indigo-700 px-5 py-2.5 text-sm font-semibold text-indigo-300 hover:bg-indigo-900/40 transition-colors"
            >
              Get Started
            </a>
          </div>
        </div>
      </section>

      {/* Production package sections */}
      <section className="px-6 py-16">
        <div className="mx-auto max-w-4xl">
          <h2 className="mb-2 text-center text-2xl font-bold text-white">
            Every Production Package Includes
          </h2>
          <p className="mb-10 text-center text-gray-400">
            All 10 sections are generated automatically for every request.
          </p>
          <div className="grid gap-4 sm:grid-cols-2">
            {sections.map((s) => (
              <div
                key={s.number}
                className="flex items-start gap-4 rounded-xl border border-gray-800 bg-gray-900 px-5 py-4"
              >
                <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-indigo-700 text-sm font-bold text-white">
                  {s.number}
                </span>
                <div>
                  <p className="font-semibold text-white">{s.name}</p>
                  <p className="text-sm text-gray-400">{s.description}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Quality labels */}
      <section className="bg-gray-900 px-6 py-16">
        <div className="mx-auto max-w-4xl">
          <h2 className="mb-2 text-center text-2xl font-bold text-white">
            Deliverable Quality Labels
          </h2>
          <p className="mb-10 text-center text-gray-400">
            Every output is labeled with exactly one honest quality descriptor.
          </p>
          <div className="overflow-hidden rounded-xl border border-gray-800">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-gray-800 bg-gray-800/60">
                  <th className="px-5 py-3 text-left font-semibold text-gray-300">
                    Label
                  </th>
                  <th className="px-5 py-3 text-left font-semibold text-gray-300">
                    Meaning
                  </th>
                </tr>
              </thead>
              <tbody>
                {qualityLabels.map((q, i) => (
                  <tr
                    key={q.label}
                    className={i % 2 === 0 ? "bg-gray-900" : "bg-gray-900/60"}
                  >
                    <td className="px-5 py-3 font-mono text-indigo-300">
                      {q.label}
                    </td>
                    <td className="px-5 py-3 text-gray-400">{q.description}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </section>

      {/* Get started */}
      <section id="get-started" className="px-6 py-16">
        <div className="mx-auto max-w-3xl">
          <h2 className="mb-2 text-center text-2xl font-bold text-white">
            Get Started
          </h2>
          <p className="mb-10 text-center text-gray-400">
            Clone the repository and run the development server in four commands.
          </p>
          <div className="rounded-xl border border-gray-800 bg-gray-900 p-6">
            <pre className="overflow-x-auto text-sm text-green-400">
              <code>{`git clone https://github.com/Alpha48Alpha/Ai-montizable-.git
cd Ai-montizable-
pnpm install
pnpm dev`}</code>
            </pre>
          </div>
          <p className="mt-6 text-center text-sm text-gray-500">
            Or generate a production package directly with Python:
          </p>
          <div className="mt-3 rounded-xl border border-gray-800 bg-gray-900 p-6">
            <pre className="text-sm text-green-400">
              <code>python movie_engine.py</code>
            </pre>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t border-gray-800 px-6 py-8 text-center text-sm text-gray-500">
        <p>
          AI-Monetizable Animation Engine — Never returns an empty or unhelpful
          response for a production request.
        </p>
      </footer>
    </main>
  );
}

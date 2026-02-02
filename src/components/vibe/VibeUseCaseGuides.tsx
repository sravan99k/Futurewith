interface UseCaseGuide {
  title: string;
  description: string;
  tools: { name: string }[];
  recommendations: string;
}

interface VibeUseCaseGuidesProps {
  guides: UseCaseGuide[];
}

export const VibeUseCaseGuides = ({ guides }: VibeUseCaseGuidesProps) => {
  return (
    <section className="py-24 bg-muted/30 border-t border-border">
      <div className="container mx-auto px-4">
        <div className="text-center mb-16">
          <h2 className="font-heading text-3xl lg:text-4xl font-black mb-4 uppercase tracking-tighter">
            Use Case <span className="text-vibe">Guides</span>
          </h2>
          <p className="text-muted-foreground font-medium max-w-xl mx-auto">
            Not sure which tools to use? Follow these recommended paths for common scenarios.
          </p>
        </div>

        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
          {guides.map((guide, idx) => (
            <div key={idx} className="p-6 rounded-2xl bg-card border border-border hover:border-vibe/40 transition-all">
              <h4 className="font-black text-lg mb-3 uppercase tracking-tight">{guide.title}</h4>
              <p className="text-sm text-muted-foreground font-medium leading-relaxed mb-4">{guide.description}</p>
              <div className="mb-4">
                <div className="text-[10px] font-black text-vibe uppercase tracking-widest mb-2">Recommended Tools</div>
                <div className="flex flex-wrap gap-2">
                  {(guide.tools || []).slice(0, 3).map((tool, tIdx) => (
                    <span key={tIdx} className="px-2 py-1 rounded bg-muted text-[10px] font-medium">
                      {tool?.name || 'Unnamed Tool'}
                    </span>
                  ))}
                </div>
              </div>
              <p className="text-xs text-muted-foreground leading-relaxed italic">
                {guide.recommendations}
              </p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
};

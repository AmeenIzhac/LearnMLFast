import { Link } from 'react-router-dom'
import './HomePage.css'

function HomePage() {
    return (
        <div className="home-page">
            {/* Hero Section */}
            <section className="hero">
                <div className="hero-content">
                    <h1 className="hero-title">Research Papers Explorer</h1>
                    <p className="hero-subtitle">
                        Discover and understand the most influential papers in AI & Machine Learning,
                        personalized to your knowledge level
                    </p>
                    <Link to="/app" className="cta-button">
                        <span>Start Exploring</span>
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                            <path d="M5 12h14M12 5l7 7-7 7" />
                        </svg>
                    </Link>
                </div>
                <div className="hero-visual">
                    <div className="floating-card card-1">
                        <span className="card-number">10K+</span>
                        <span>Research Papers</span>
                    </div>
                    <div className="floating-card card-2">
                        <span className="card-number">10</span>
                        <span>Top Venues</span>
                    </div>
                    <div className="floating-card card-3">
                        <span className="card-number">AI</span>
                        <span>Powered Insights</span>
                    </div>
                </div>
            </section>

            {/* Features Section */}
            <section className="features">
                <h2 className="section-title">How It Works</h2>
                <div className="features-grid">
                    <div className="feature-card">
                        <div className="feature-icon">
                            <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                <path d="M9 19c-5 1.5-5-2.5-7-3m14 6v-3.87a3.37 3.37 0 0 0-.94-2.61c3.14-.35 6.44-1.54 6.44-7A5.44 5.44 0 0 0 20 4.77 5.07 5.07 0 0 0 19.91 1S18.73.65 16 2.48a13.38 13.38 0 0 0-7 0C6.27.65 5.09 1 5.09 1A5.07 5.07 0 0 0 5 4.77a5.44 5.44 0 0 0-1.5 3.78c0 5.42 3.3 6.61 6.44 7A3.37 3.37 0 0 0 9 18.13V22" />
                            </svg>
                        </div>
                        <h3>Calibrate Your Knowledge</h3>
                        <p>
                            Answer a few quick questions about ML concepts.
                            We adapt to your expertise level—from beginner to researcher.
                        </p>
                    </div>

                    <div className="feature-card">
                        <div className="feature-icon">
                            <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                <circle cx="11" cy="11" r="8" />
                                <line x1="21" y1="21" x2="16.65" y2="16.65" />
                            </svg>
                        </div>
                        <h3>Browse Top Papers</h3>
                        <p>
                            Explore papers from NeurIPS, ICML, CVPR, ACL and more.
                            Filter by venue, sort by citations, and find what matters.
                        </p>
                    </div>

                    <div className="feature-card">
                        <div className="feature-icon">
                            <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                <path d="M12 2L2 7l10 5 10-5-10-5z" />
                                <path d="M2 17l10 5 10-5" />
                                <path d="M2 12l10 5 10-5" />
                            </svg>
                        </div>
                        <h3>Make It Intuitive</h3>
                        <p>
                            Click "Make Intuitive" on any paper.
                            Our AI rewrites the abstract at your level, explaining concepts you don't know.
                        </p>
                    </div>

                    <div className="feature-card">
                        <div className="feature-icon">
                            <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
                            </svg>
                        </div>
                        <h3>Ask Questions</h3>
                        <p>
                            Still confused? Chat with AI about the paper.
                            It remembers what you struggled with and improves explanations over time.
                        </p>
                    </div>
                </div>
            </section>

            {/* Venues Section */}
            <section className="venues-section">
                <h2 className="section-title">Top AI Venues Covered</h2>
                <div className="venues-list">
                    <div className="venue-badge neurips">NeurIPS</div>
                    <div className="venue-badge icml">ICML</div>
                    <div className="venue-badge iclr">ICLR</div>
                    <div className="venue-badge cvpr">CVPR</div>
                    <div className="venue-badge iccv">ICCV</div>
                    <div className="venue-badge acl">ACL</div>
                    <div className="venue-badge emnlp">EMNLP</div>
                    <div className="venue-badge aaai">AAAI</div>
                </div>
            </section>

            {/* CTA Section */}
            <section className="cta-section">
                <h2>Ready to Learn Smarter?</h2>
                <p>No signup required. Just calibrate and start exploring.</p>
                <Link to="/app" className="cta-button cta-large">
                    <span>Open the Explorer</span>
                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <path d="M5 12h14M12 5l7 7-7 7" />
                    </svg>
                </Link>
            </section>

            {/* Footer */}
            <footer className="footer">
                <p>Built for researchers, students, and curious minds.</p>
            </footer>
        </div>
    )
}

export default HomePage

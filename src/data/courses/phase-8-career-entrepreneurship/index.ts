import { Phase, TopicFolder } from '../types';

const folders: TopicFolder[] = [
    {
        "id": "folder-8-00_domain_specialization",
        "name": "domain-specialization",
        "description": "Domain Specialization and Career Growth",
        "topics": [
            {
                "id": "8-00-domain-0",
                "title": "Domain Specialization Guide",
                "description": "Learn how to specialize in your chosen domain",
                "duration": "1-2 hours",
                "type": "theory",
                "markdownPath": "/courses/phase-8-career-entrepreneurship/domain-specialization/README.md"
            },
            {
                "id": "8-00-domain-1",
                "title": "Step-by-Step Domain Specialization",
                "description": "Practical steps to achieve domain expertise",
                "duration": "1-2 hours",
                "type": "practice",
                "markdownPath": "/courses/phase-8-career-entrepreneurship/domain-specialization/step1-domain-specialization.md"
            }
        ]
    },
    {
        "id": "folder-8-01_ai_job_market",
        "name": "01_ai_job_market",
        "description": "Topics for 01_ai_job_market",
        "topics": [
            {
                "id": "8-01_ai_job_market-0",
                "title": "Competitive Platforms Analysis",
                "description": "Learning about 01_competitive_platforms_analysis",
                "duration": "1-2 hours",
                "type": "theory",
                "markdownPath": "/courses/phase-8-career-entrepreneurship/01_ai_job_market/01_competitive_platforms_analysis.md"
            },
            {
                "id": "8-01_ai_job_market-1",
                "title": "Job Market Salary Breakdown",
                "description": "Learning about 02_job_market_salary_breakdown",
                "duration": "1-2 hours",
                "type": "theory",
                "markdownPath": "/courses/phase-8-career-entrepreneurship/01_ai_job_market/02_job_market_salary_breakdown.md"
            },
            {
                "id": "8-01_ai_job_market-2",
                "title": "Gap Analysis By Country",
                "description": "Learning about 03_gap_analysis_by_country",
                "duration": "1-2 hours",
                "type": "theory",
                "markdownPath": "/courses/phase-8-career-entrepreneurship/01_ai_job_market/03_gap_analysis_by_country.md"
            },
            {
                "id": "8-01_ai_job_market-3",
                "title": "Curriculum Redundancy Report",
                "description": "Learning about 04_curriculum_redundancy_report",
                "duration": "1-2 hours",
                "type": "theory",
                "markdownPath": "/courses/phase-8-career-entrepreneurship/01_ai_job_market/04_curriculum_redundancy_report.md"
            },
            {
                "id": "8-01_ai_job_market-4",
                "title": "Step1 JOB MARKET COMPLETE GUIDE",
                "description": "Learning about step1-JOB_MARKET_COMPLETE_GUIDE",
                "duration": "1-2 hours",
                "type": "theory",
                "markdownPath": "/courses/phase-8-career-entrepreneurship/01_ai_job_market/step1-JOB_MARKET_COMPLETE_GUIDE.md"
            }
        ]
    },
    {
        "id": "folder-8-02_ai_entrepreneurship",
        "name": "02_ai_entrepreneurship",
        "description": "Topics for 02_ai_entrepreneurship",
        "topics": [
            {
                "id": "8-02_ai_entrepreneurship-0",
                "title": "Step1 Entrepreneurship COMPLETE GUIDE",
                "description": "Learning about step1-entrepreneurship_COMPLETE_GUIDE",
                "duration": "1-2 hours",
                "type": "theory",
                "markdownPath": "/courses/phase-8-career-entrepreneurship/02_ai_entrepreneurship/step1-entrepreneurship_COMPLETE_GUIDE.md"
            },
            {
                "id": "8-02_ai_entrepreneurship-1",
                "title": "Step2 Entrepreneurship Interview Prep",
                "description": "Learning about step2-entrepreneurship_interview_prep",
                "duration": "1-2 hours",
                "type": "theory",
                "markdownPath": "/courses/phase-8-career-entrepreneurship/02_ai_entrepreneurship/step2-entrepreneurship_interview_prep.md"
            }
        ]
    },
    {
        "id": "folder-8-03_freelancing_startups",
        "name": "03_freelancing_startups",
        "description": "Topics for 03_freelancing_startups",
        "topics": [
            {
                "id": "8-03_freelancing_startups-0",
                "title": "Step1 Freelancing Roadmap Theory",
                "description": "Learning about step1-freelancing_roadmap_theory",
                "duration": "1-2 hours",
                "type": "theory",
                "markdownPath": "/courses/phase-8-career-entrepreneurship/03_freelancing_startups/step1-freelancing_roadmap_theory.md"
            },
            {
                "id": "8-03_freelancing_startups-1",
                "title": "Step2 Startup Case Studies",
                "description": "Learning about step2-startup_case_studies",
                "duration": "1-2 hours",
                "type": "theory",
                "markdownPath": "/courses/phase-8-career-entrepreneurship/03_freelancing_startups/step2-startup_case_studies.md"
            },
            {
                "id": "8-03_freelancing_startups-2",
                "title": "Step3 Freelancing Cheatsheet",
                "description": "Learning about step3-freelancing_cheatsheet",
                "duration": "1-2 hours",
                "type": "theory",
                "markdownPath": "/courses/phase-8-career-entrepreneurship/03_freelancing_startups/step3-freelancing_cheatsheet.md"
            }
        ]
    }
];

const allTopics = folders.flatMap(folder => folder.topics);

export const phase8: Phase = {
  id: 'phase-8',
  number: 8,
  title: 'Career & Entrepreneurship',
  description: 'Navigate the AI job market and build your own AI-powered business',
  duration: '4-6 weeks',
  skills: ["Job Search","Entrepreneurship","Freelancing","Networking"],
  topics: allTopics,
  folders: folders
};

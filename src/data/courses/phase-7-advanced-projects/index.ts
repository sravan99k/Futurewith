import { Phase, TopicFolder } from '../types';

const folders: TopicFolder[] = [
    {
        "id": "folder-7-01-capstone-planning",
        "name": "01-capstone-planning",
        "description": "Topics for 01-capstone-planning",
        "topics": [
            {
                "id": "7-01-0",
                "title": "Step1 Capstone Planning",
                "description": "Learning about step1-capstone-planning",
                "duration": "1-2 hours",
                "type": "theory",
                "markdownPath": "/courses/phase-7-advanced-projects/01-capstone-planning/step1-capstone-planning.md"
            }
        ]
    },
    {
        "id": "folder-7-03-legal-contract-analysis",
        "name": "03-legal-contract-analysis",
        "description": "Topics for 03-legal-contract-analysis",
        "topics": [
            {
                "id": "7-03-legal-0",
                "title": "README Legal Contract Analysis",
                "description": "Learning about 03-legal-contract-analysis",
                "duration": "1-2 hours",
                "type": "theory",
                "markdownPath": "/courses/phase-7-advanced-projects/03-legal-contract-analysis/README.md"
            }
        ]
    },
    {
        "id": "folder-7-02-fullstack-ai-application",
        "name": "02-fullstack-ai-application",
        "description": "Topics for 02-fullstack-ai-application",
        "topics": [
            {
                "id": "7-02-0",
                "title": "Step1 Fullstack Ai Application",
                "description": "Learning about step1-fullstack-ai-application",
                "duration": "1-2 hours",
                "type": "theory",
                "markdownPath": "/courses/phase-7-advanced-projects/02-fullstack-ai-application/step1-fullstack-ai-application.md"
            }
        ]
    },
    {
        "id": "folder-7-04-predictive-maintenance",
        "name": "04-predictive-maintenance",
        "description": "Topics for 04-predictive-maintenance",
        "topics": [
            {
                "id": "7-04-maintenance-0",
                "title": "README Predictive Maintenance",
                "description": "Learning about 04-predictive-maintenance",
                "duration": "1-2 hours",
                "type": "theory",
                "markdownPath": "/courses/phase-7-advanced-projects/04-predictive-maintenance/README.md"
            }
        ]
    },
    {
        "id": "folder-7-06-autonomous-customer-support",
        "name": "06-autonomous-customer-support",
        "description": "Topics for 06-autonomous-customer-support",
        "topics": [
            {
                "id": "7-06-support-0",
                "title": "README Autonomous Customer Support",
                "description": "Learning about 06-autonomous-customer-support",
                "duration": "1-2 hours",
                "type": "theory",
                "markdownPath": "/courses/phase-7-advanced-projects/06-autonomous-customer-support/README.md"
            }
        ]
    },
    {
        "id": "folder-7-07-financial-portfolio-risk",
        "name": "07-financial-portfolio-risk",
        "description": "Topics for 07-financial-portfolio-risk",
        "topics": [
            {
                "id": "7-07-finance-0",
                "title": "README Financial Portfolio Risk",
                "description": "Learning about 07-financial-portfolio-risk",
                "duration": "1-2 hours",
                "type": "theory",
                "markdownPath": "/courses/phase-7-advanced-projects/07-financial-portfolio-risk/README.md"
            }
        ]
    },
    {
        "id": "folder-7-08-recruitment-platform",
        "name": "08-recruitment-platform",
        "description": "Topics for 08-recruitment-platform",
        "topics": [
            {
                "id": "7-08-recruitment-0",
                "title": "README Recruitment Platform",
                "description": "Learning about 08-recruitment-platform",
                "duration": "1-2 hours",
                "type": "theory",
                "markdownPath": "/courses/phase-7-advanced-projects/08-recruitment-platform/README.md"
            }
        ]
    },
    {
        "id": "folder-7-09-climate-risk-esg",
        "name": "09-climate-risk-esg",
        "description": "Topics for 09-climate-risk-esg",
        "topics": [
            {
                "id": "7-09-esg-0",
                "title": "README Climate Risk & ESG",
                "description": "Learning about 09-climate-risk-esg",
                "duration": "1-2 hours",
                "type": "theory",
                "markdownPath": "/courses/phase-7-advanced-projects/09-climate-risk-esg/README.md"
            }
        ]
    }
];

const parseFolderOrder = (folderName: string) => {
  const prefix = folderName.split('-')[0];
  const parsed = Number.parseInt(prefix, 10);
  return Number.isFinite(parsed) ? parsed : Number.POSITIVE_INFINITY;
};

folders.sort((a, b) => {
  const aOrder = parseFolderOrder(a.name);
  const bOrder = parseFolderOrder(b.name);
  if (aOrder !== bOrder) return aOrder - bOrder;
  return a.name.localeCompare(b.name);
});

const allTopics = folders.flatMap(folder => folder.topics);

export const phase7: Phase = {
  id: 'phase-7',
  number: 7,
  title: 'Advanced Projects & Specializations',
  description: 'Build capstone projects, specialize in your chosen domain, and create a professional portfolio',
  duration: '6-8 weeks',
  skills: ["Full-Stack Projects","Domain Specialization","Portfolio","Open Source"],
  topics: allTopics,
  folders: folders
};

import { useState } from 'react'
import { Button } from '@/components/ui/button.jsx'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card.jsx'
import { Badge } from '@/components/ui/badge.jsx'
import { Avatar, AvatarFallback, AvatarImage } from '@/components/ui/avatar.jsx'
import { Progress } from '@/components/ui/progress.jsx'
import ChatbotWidget from './components/ChatbotWidget.jsx'
import { 
  BookOpen, 
  FileText, 
  Users, 
  Settings, 
  Bell, 
  Search, 
  Plus,
  BarChart3,
  Clock,
  CheckCircle,
  AlertCircle,
  Menu,
  Home,
  Upload,
  Eye,
  Edit,
  Calendar
} from 'lucide-react'
import './App.css'

function App() {
  const [activeTab, setActiveTab] = useState('dashboard')
  const [sidebarOpen, setSidebarOpen] = useState(true)

  // Mock data for demonstration
  const userProfile = {
    name: "Dr. Sarah Johnson",
    role: "Author & Reviewer",
    avatar: "/api/placeholder/40/40",
    initials: "SJ"
  }

  const dashboardStats = {
    totalSubmissions: 12,
    activeSubmissions: 3,
    acceptedSubmissions: 7,
    rejectedSubmissions: 2,
    pendingReviews: 2,
    completedReviews: 15
  }

  const recentSubmissions = [
    {
      id: 1,
      title: "Machine Learning Applications in Academic Publishing",
      status: "Under Review",
      stage: "Peer Review",
      submittedDate: "2024-01-15",
      progress: 65
    },
    {
      id: 2,
      title: "Digital Transformation in Scholarly Communication",
      status: "Revision Required",
      stage: "Author Revision",
      submittedDate: "2024-01-10",
      progress: 40
    },
    {
      id: 3,
      title: "Open Access Publishing Models: A Comparative Study",
      status: "Accepted",
      stage: "Production",
      submittedDate: "2023-12-20",
      progress: 90
    }
  ]

  const upcomingDeadlines = [
    {
      type: "Review",
      description: "Review for 'AI in Education Research'",
      dueDate: "2024-01-25",
      priority: "high"
    },
    {
      type: "Revision",
      description: "Submit revised manuscript",
      dueDate: "2024-01-30",
      priority: "medium"
    }
  ]

  const navigationItems = [
    { id: 'dashboard', label: 'Dashboard', icon: Home },
    { id: 'submissions', label: 'My Submissions', icon: FileText },
    { id: 'reviews', label: 'Review Assignments', icon: Eye },
    { id: 'journals', label: 'Journals', icon: BookOpen },
    { id: 'profile', label: 'Profile', icon: Users },
    { id: 'settings', label: 'Settings', icon: Settings }
  ]

  const getStatusColor = (status) => {
    switch (status.toLowerCase()) {
      case 'accepted': return 'bg-green-100 text-green-800'
      case 'under review': return 'bg-blue-100 text-blue-800'
      case 'revision required': return 'bg-yellow-100 text-yellow-800'
      case 'rejected': return 'bg-red-100 text-red-800'
      default: return 'bg-gray-100 text-gray-800'
    }
  }

  const getPriorityColor = (priority) => {
    switch (priority) {
      case 'high': return 'border-l-red-500'
      case 'medium': return 'border-l-yellow-500'
      case 'low': return 'border-l-green-500'
      default: return 'border-l-gray-500'
    }
  }

  return (
    <div className="min-h-screen bg-background">
      {/* Top Navigation */}
      <header className="border-b bg-card shadow-sm">
        <div className="flex h-16 items-center px-4">
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setSidebarOpen(!sidebarOpen)}
            className="mr-4 lg:hidden"
          >
            <Menu className="h-5 w-5" />
          </Button>
          
          <div className="flex items-center space-x-4">
            <BookOpen className="h-8 w-8 text-primary" />
            <div>
              <h1 className="text-xl font-bold text-foreground">Open Journal Systems</h1>
              <p className="text-sm text-muted-foreground">Modern Publishing Platform</p>
            </div>
          </div>

          <div className="ml-auto flex items-center space-x-4">
            <div className="relative hidden md:block">
              <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
              <input
                type="search"
                placeholder="Search submissions, journals..."
                className="w-64 rounded-md border border-input bg-background pl-10 pr-4 py-2 text-sm focus:border-ring focus:outline-none focus:ring-2 focus:ring-ring/20"
              />
            </div>
            
            <Button variant="ghost" size="sm">
              <Bell className="h-5 w-5" />
            </Button>
            
            <Avatar>
              <AvatarImage src={userProfile.avatar} alt={userProfile.name} />
              <AvatarFallback>{userProfile.initials}</AvatarFallback>
            </Avatar>
          </div>
        </div>
      </header>

      <div className="flex">
        {/* Sidebar */}
        <aside className={`${sidebarOpen ? 'w-64' : 'w-0'} transition-all duration-300 overflow-hidden border-r bg-card`}>
          <nav className="p-4 space-y-2">
            {navigationItems.map((item) => {
              const Icon = item.icon
              return (
                <Button
                  key={item.id}
                  variant={activeTab === item.id ? "default" : "ghost"}
                  className="w-full justify-start"
                  onClick={() => setActiveTab(item.id)}
                >
                  <Icon className="mr-3 h-4 w-4" />
                  {item.label}
                </Button>
              )
            })}
          </nav>
        </aside>

        {/* Main Content */}
        <main className="flex-1 p-6">
          {activeTab === 'dashboard' && (
            <div className="space-y-6">
              {/* Welcome Section */}
              <div className="flex items-center justify-between">
                <div>
                  <h2 className="text-3xl font-bold text-foreground">Welcome back, {userProfile.name.split(' ')[1]}</h2>
                  <p className="text-muted-foreground">Here's what's happening with your submissions and reviews</p>
                </div>
                <Button className="bg-primary hover:bg-primary/90">
                  <Plus className="mr-2 h-4 w-4" />
                  New Submission
                </Button>
              </div>

              {/* Stats Cards */}
              <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
                <Card>
                  <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                    <CardTitle className="text-sm font-medium">Total Submissions</CardTitle>
                    <FileText className="h-4 w-4 text-muted-foreground" />
                  </CardHeader>
                  <CardContent>
                    <div className="text-2xl font-bold">{dashboardStats.totalSubmissions}</div>
                    <p className="text-xs text-muted-foreground">+2 from last month</p>
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                    <CardTitle className="text-sm font-medium">Active Submissions</CardTitle>
                    <Clock className="h-4 w-4 text-muted-foreground" />
                  </CardHeader>
                  <CardContent>
                    <div className="text-2xl font-bold">{dashboardStats.activeSubmissions}</div>
                    <p className="text-xs text-muted-foreground">Currently in review</p>
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                    <CardTitle className="text-sm font-medium">Accepted Papers</CardTitle>
                    <CheckCircle className="h-4 w-4 text-muted-foreground" />
                  </CardHeader>
                  <CardContent>
                    <div className="text-2xl font-bold">{dashboardStats.acceptedSubmissions}</div>
                    <p className="text-xs text-muted-foreground">58% acceptance rate</p>
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                    <CardTitle className="text-sm font-medium">Pending Reviews</CardTitle>
                    <AlertCircle className="h-4 w-4 text-muted-foreground" />
                  </CardHeader>
                  <CardContent>
                    <div className="text-2xl font-bold">{dashboardStats.pendingReviews}</div>
                    <p className="text-xs text-muted-foreground">Due this week</p>
                  </CardContent>
                </Card>
              </div>

              {/* Recent Submissions */}
              <Card>
                <CardHeader>
                  <CardTitle>Recent Submissions</CardTitle>
                  <CardDescription>Track the progress of your latest submissions</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    {recentSubmissions.map((submission) => (
                      <div key={submission.id} className="flex items-center space-x-4 p-4 border rounded-lg">
                        <div className="flex-1 space-y-2">
                          <div className="flex items-center justify-between">
                            <h4 className="font-medium">{submission.title}</h4>
                            <Badge className={getStatusColor(submission.status)}>
                              {submission.status}
                            </Badge>
                          </div>
                          <div className="flex items-center space-x-4 text-sm text-muted-foreground">
                            <span>Stage: {submission.stage}</span>
                            <span>•</span>
                            <span>Submitted: {submission.submittedDate}</span>
                          </div>
                          <div className="space-y-1">
                            <div className="flex justify-between text-sm">
                              <span>Progress</span>
                              <span>{submission.progress}%</span>
                            </div>
                            <Progress value={submission.progress} className="h-2" />
                          </div>
                        </div>
                        <div className="flex space-x-2">
                          <Button variant="outline" size="sm">
                            <Eye className="h-4 w-4" />
                          </Button>
                          <Button variant="outline" size="sm">
                            <Edit className="h-4 w-4" />
                          </Button>
                        </div>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>

              {/* Upcoming Deadlines */}
              <Card>
                <CardHeader>
                  <CardTitle>Upcoming Deadlines</CardTitle>
                  <CardDescription>Don't miss these important dates</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-3">
                    {upcomingDeadlines.map((deadline, index) => (
                      <div key={index} className={`flex items-center space-x-4 p-3 border-l-4 ${getPriorityColor(deadline.priority)} bg-muted/50 rounded-r-lg`}>
                        <Calendar className="h-5 w-5 text-muted-foreground" />
                        <div className="flex-1">
                          <p className="font-medium">{deadline.description}</p>
                          <p className="text-sm text-muted-foreground">Due: {deadline.dueDate}</p>
                        </div>
                        <Badge variant="outline">{deadline.type}</Badge>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            </div>
          )}

          {activeTab !== 'dashboard' && (
            <div className="flex items-center justify-center h-64">
              <div className="text-center">
                <h3 className="text-lg font-medium text-foreground mb-2">
                  {navigationItems.find(item => item.id === activeTab)?.label}
                </h3>
                <p className="text-muted-foreground">This section is under development</p>
              </div>
            </div>
          )}
        </main>
      </div>

      {/* Chatbot Widget */}
      <ChatbotWidget 
        userContext={{
          userId: 1,
          page: activeTab,
          userRole: userProfile.role.toLowerCase().includes('author') ? 'author' : 'editor',
          userName: userProfile.name
        }}
      />
    </div>
  )
}

export default App


import { useState } from 'react';
import Header from './components/Header/Header';
import Footer from './components/Footer/Footer';
import TabBar from './components/common/TabBar';
import POIManagement from './components/POI/POIManagement';
import LiveAlerts from './components/Alerts/LiveAlerts';
import SearchPanel from './components/Search/SearchPanel';

const TABS = ['POI Management', 'Live Alerts', 'Search History'];

const App = () => {
  const [activeTab, setActiveTab] = useState(TABS[0]);

  return (
    <div className="flex flex-col h-screen bg-gray-50 font-body text-intel-dark">
      <Header />
      <TabBar tabs={TABS} activeTab={activeTab} onTabChange={setActiveTab} />

      <main className="flex-1 overflow-hidden">
        <div className={`h-full ${activeTab === 'POI Management' ? '' : 'hidden'}`}>
          <POIManagement />
        </div>
        <div className={`h-full ${activeTab === 'Live Alerts' ? '' : 'hidden'}`}>
          <LiveAlerts />
        </div>
        <div className={`h-full ${activeTab === 'Search History' ? '' : 'hidden'}`}>
          <SearchPanel />
        </div>
      </main>

      <Footer />
    </div>
  );
};

export default App;

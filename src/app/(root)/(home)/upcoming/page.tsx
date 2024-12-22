import CallList from '@/components/meet-components/CallList';

const UpcomingPage = () => {
  return (
    <section className="flex size-full flex-col gap-10 text-black">
      <h1 className="text-3xl font-bold">Upcoming Meeting</h1>

      <CallList type="upcoming" />
    </section>
  );
};

export default UpcomingPage;
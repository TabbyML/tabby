export default function RootLayout({
  children
}: {
  children: React.ReactNode
}) {
  return (
    <div className="flex flex-1 flex-col items-center justify-center">
      {children}
    </div>
  )
}

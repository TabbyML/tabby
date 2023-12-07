export default function RootLayout({
  children
}: {
  children: React.ReactNode
}) {
  return (
    <div className="flex flex-col items-center justify-center flex-1">
      {children}
    </div>
  )
}

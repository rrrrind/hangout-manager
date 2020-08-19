class AddDetailsToHangOuts < ActiveRecord::Migration[6.0]
  def change
    add_column :hang_outs, :label, :integer, default: 0
    add_column :hang_outs, :upper_member, :integer, default: 1
  end
end
